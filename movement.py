import math
from datetime import datetime
from pathlib import Path
import numpy as np
from PIL import ImageStat
from tdw.add_ons.image_capture import ImageCapture

class Agent:
    DEFAULT_MOVE_STEP = 1.0
    DEFAULT_TURN_ANGLE = 45
    DEFAULT_LOOK_ANGLE = 15
    DEFAULT_MIN_CAMERA_PITCH = -30
    DEFAULT_MAX_CAMERA_PITCH = 30

    def __init__(
        self,
        controller,
        avatar_id,
        initial_position,
        scene,
        initial_yaw=0,
        initial_camera_pitch=0,
        camera_pitch_step=DEFAULT_LOOK_ANGLE,
        minimum_camera_pitch=DEFAULT_MIN_CAMERA_PITCH,
        maximum_camera_pitch=DEFAULT_MAX_CAMERA_PITCH,
        turn_angle=DEFAULT_TURN_ANGLE,
        blank_frame_retry_count=2,
        reject_blank_frames=True,
    ):
        self.controller = controller
        self.avatar_id = avatar_id
        self.position = initial_position.copy()
        self.yaw = float(initial_yaw)
        self.camera_yaw = 0
        self.camera_pitch_step = float(camera_pitch_step)
        self.minimum_camera_pitch = float(minimum_camera_pitch)
        self.maximum_camera_pitch = float(maximum_camera_pitch)
        if self.camera_pitch_step <= 0:
            raise ValueError("camera_pitch_step must be positive.")
        if self.minimum_camera_pitch > self.maximum_camera_pitch:
            raise ValueError(
                "minimum_camera_pitch cannot exceed maximum_camera_pitch."
            )
        self.initial_camera_pitch = float(initial_camera_pitch)
        if not (
            self.minimum_camera_pitch
            <= self.initial_camera_pitch
            <= self.maximum_camera_pitch
        ):
            raise ValueError(
                "initial_camera_pitch must be within the configured camera "
                "pitch bounds."
            )
        self.camera_pitch = self.initial_camera_pitch
        self.scene = scene
        self.move_step = self.DEFAULT_MOVE_STEP
        self.turn_angle = float(turn_angle)
        if (
            not math.isfinite(self.turn_angle)
            or self.turn_angle <= 0
            or not math.isclose(
                360.0 / self.turn_angle,
                round(360.0 / self.turn_angle),
                rel_tol=0,
                abs_tol=1e-6,
            )
        ):
            raise ValueError(
                "turn_angle must be positive and divide 360 degrees evenly."
            )
        self.wall_margin = 0.5
        self.collision_radius = 0.4
        self.held_object = None
        self.interaction_distance = 5.0
        self.capture = None
        self.capture_directory = Path("captures")
        self.blank_frame_retry_count = max(0, int(blank_frame_retry_count))
        self.reject_blank_frames = bool(reject_blank_frames)
        self.last_frame_quality = None
        self._segmentation_color_cache = {}

    def setup_avatar(self):
        yaw_rad = math.radians(self.yaw)
        commands = [
            {"$type": "create_avatar",
             "type": "A_Img_Caps_Kinematic",
             "id": self.avatar_id},
            {"$type": "teleport_avatar_to",
             "avatar_id": self.avatar_id,
             "position": self.position},
            {"$type": "set_avatar_forward",
             "avatar_id": self.avatar_id,
             "forward": {
                 "x": math.sin(yaw_rad),
                 "y": 0,
                 "z": math.cos(yaw_rad),
             }},
            {"$type": "set_render_quality",
             "render_quality": 5}
        ]
        if self.camera_pitch:
            commands.append(
                {
                    "$type": "rotate_sensor_container_by",
                    "avatar_id": self.avatar_id,
                    "axis": "pitch",
                    "angle": self.camera_pitch,
                }
            )
        self.controller.communicate(commands)
        print("✓ Avatar created")

    def check_collision(self, new_x, new_z):
        path_navigable = None
        if hasattr(self.scene, "is_path_navigable"):
            path_navigable = self.scene.is_path_navigable(
                self.position["x"],
                self.position["z"],
                new_x,
                new_z,
            )
        if path_navigable is False:
            return True, "occupancy_map"

        if path_navigable is None:
            room_size = self.scene.get_room_size()
            half_room = room_size / 2.0
            if (
                new_x > half_room - self.wall_margin
                or new_x < -half_room + self.wall_margin
                or new_z > half_room - self.wall_margin
                or new_z < -half_room + self.wall_margin
            ):
                return True, "wall"

        object_positions = self.scene.get_object_positions()
        for obj in object_positions:
            if self.held_object and obj.get("object_id") == self.held_object["object_id"]:
                continue

            distance = math.sqrt((new_x - obj["x"])**2 + (new_z - obj["z"])**2)
            if distance < (self.collision_radius + obj["radius"]):
                return True, "object"

        return False, None

    def get_collision_safe_actions(self, allowed_actions=None):
        default_actions = (
            "move_forward",
            "move_backward",
            "turn_left",
            "turn_right",
            "look_up",
            "look_down",
            "pick_up",
            "put_down",
        )
        requested = (
            list(allowed_actions)
            if allowed_actions is not None
            else list(default_actions)
        )
        safe_actions = []
        yaw_rad = math.radians(self.yaw)
        for action in requested:
            if action == "move_forward":
                new_x = self.position["x"] + self.move_step * math.sin(yaw_rad)
                new_z = self.position["z"] + self.move_step * math.cos(yaw_rad)
                collision, _ = self.check_collision(new_x, new_z)
                if collision:
                    continue
            elif action == "move_backward":
                new_x = self.position["x"] - self.move_step * math.sin(yaw_rad)
                new_z = self.position["z"] - self.move_step * math.cos(yaw_rad)
                collision, _ = self.check_collision(new_x, new_z)
                if collision:
                    continue
            elif (
                action == "look_up"
                and self.camera_pitch <= self.minimum_camera_pitch
            ):
                continue
            elif (
                action == "look_down"
                and self.camera_pitch >= self.maximum_camera_pitch
            ):
                continue
            safe_actions.append(action)
        return safe_actions

    def get_nearby_objects(self):
        nearby = []
        object_positions = self.scene.get_object_positions()

        print(f"\n[DEBUG] Checking from position: x={self.position['x']:.2f}, z={self.position['z']:.2f}")
        print(f"[DEBUG] Total objects in scene: {len(object_positions)}")

        for obj in object_positions:
            distance = math.sqrt(
                (self.position['x'] - obj["x"])**2 +
                (self.position['z'] - obj["z"])**2
            )

            pickupable = obj.get("pickupable", False)
            print(f"[DEBUG] {obj['name']} (ID: {obj['object_id']}) - distance: {distance:.2f}m, pickupable: {pickupable}")

            if distance <= self.interaction_distance and pickupable:
                nearby.append({
                    "object_id": obj["object_id"],
                    "name": obj["name"],
                    "distance": distance
                })

        return sorted(nearby, key=lambda x: x["distance"])

    def pick_up(self):
        if self.held_object is not None:
            print(f"⚠️  Already holding {self.held_object['name']}! Put it down first.")
            return False

        nearby = self.get_nearby_objects()

        if not nearby:
            print(f"⚠️  No objects within reach ({self.interaction_distance}m) to pick up!")
            return False

        target = nearby[0]
        object_id = target["object_id"]

        commands = [
            {"$type": "set_kinematic_state",
             "id": object_id,
             "is_kinematic": True,
             "use_gravity": False},
            {"$type": "parent_object_to_avatar",
             "avatar_id": self.avatar_id,
             "id": object_id},
            {"$type": "teleport_object",
             "id": object_id,
             "position": {"x": 0.3, "y": -0.3, "z": 0.5}}
        ]

        self.controller.communicate(commands)

        self.held_object = {
            "object_id": object_id,
            "name": target["name"]
        }

        self._render()
        print(f"✓ Picked up {target['name']} (ID: {object_id})")
        return True

    def put_down(self):
        if self.held_object is None:
            print("⚠️  Not holding anything!")
            return False

        object_id = self.held_object["object_id"]

        yaw_rad = math.radians(self.yaw)
        put_down_distance = 1.5

        new_x = self.position['x'] + put_down_distance * math.sin(yaw_rad)
        new_z = self.position['z'] + put_down_distance * math.cos(yaw_rad)
        new_y = 1.0

        commands = [
            {"$type": "unparent_object",
             "id": object_id},
            {"$type": "teleport_object",
             "id": object_id,
             "position": {"x": new_x, "y": new_y, "z": new_z}},
            {"$type": "set_kinematic_state",
             "id": object_id,
             "is_kinematic": False,
             "use_gravity": True}
        ]

        self.controller.communicate(commands)

        for i in range(30):
            self.controller.communicate([])

        self.scene.update_object_position(object_id, new_x, new_z)

        print(f"✓ Put down {self.held_object['name']}")

        self.held_object = None
        return True

    def setup_image_capture(self):
        self.capture_directory.mkdir(
            parents=True,
            exist_ok=True
        )

        self.capture = ImageCapture(
            path=self.capture_directory,
            avatar_ids=[self.avatar_id],
            png=True
        )

        self.controller.add_ons.append(self.capture)

        self.capture.set(frequency="never")

    def get_current_view(self):
        if self.capture is None:
            raise RuntimeError(
                "Image capture has not been set up."
            )

        attempts = self.blank_frame_retry_count + 1
        last_image = None
        for attempt in range(1, attempts + 1):
            self.capture.set(
                frequency="once",
                avatar_ids=[self.avatar_id],
                pass_masks=["_img"],
                save=False
            )

            self.controller.communicate([])
            images = self.capture.get_pil_images()

            if self.avatar_id not in images:
                if attempt == attempts:
                    raise RuntimeError(
                        f"No image returned for avatar '{self.avatar_id}'."
                    )
                continue

            avatar_images = images[self.avatar_id]
            if "_img" not in avatar_images:
                if attempt == attempts:
                    raise RuntimeError("No RGB image was returned.")
                continue

            last_image = avatar_images["_img"].copy()
            self.last_frame_quality = self._frame_quality(last_image)
            self.last_frame_quality["capture_attempt"] = attempt
            if not self.last_frame_quality["appears_blank"]:
                return last_image

        if self.reject_blank_frames:
            raise RuntimeError(
                "TDW returned an unusable near-uniform black or white camera "
                f"frame after {attempts} attempt(s): {self.last_frame_quality}. "
                "Check the initial pose and occupancy-map navigation."
            )
        return last_image

    def get_render_pass(self, pass_mask):
        if self.capture is None:
            raise RuntimeError(
                "Image capture has not been set up."
            )
        normalized_mask = str(pass_mask).strip()
        if not normalized_mask:
            raise ValueError("pass_mask must be non-empty.")
        self.capture.set(
            frequency="once",
            avatar_ids=[self.avatar_id],
            pass_masks=[normalized_mask],
            save=False,
        )
        self.controller.communicate([])
        images = self.capture.get_pil_images()
        avatar_images = images.get(self.avatar_id, {})
        image = avatar_images.get(normalized_mask)
        if image is None:
            raise RuntimeError(
                f"No {normalized_mask!r} image was returned for avatar "
                f"{self.avatar_id!r}."
            )
        self.capture.set(frequency="never")
        return image.copy()

    def _segmentation_color(self, object_id):
        object_id = int(object_id)
        if object_id in self._segmentation_color_cache:
            return self._segmentation_color_cache[object_id]

        from tdw.output_data import OutputData, SegmentationColors

        response = self.controller.communicate(
            {"$type": "send_segmentation_colors", "frequency": "once"}
        )
        color = None
        for item in response:
            if OutputData.get_data_type_id(item) != "segm":
                continue
            segmentation = SegmentationColors(item)
            for index in range(segmentation.get_num()):
                if segmentation.get_object_id(index) != object_id:
                    continue
                color = tuple(
                    int(value)
                    for value in list(segmentation.get_object_color(index))[:3]
                )
                break
            if color is not None:
                break
        if color is not None:
            self._segmentation_color_cache[object_id] = color
        return color

    @staticmethod
    def _count_instance_pixels(image, color):
        pixels = np.asarray(image.convert("RGB"), dtype=np.uint8)
        expected = np.asarray(color, dtype=np.uint8)
        return int(np.count_nonzero(np.all(pixels == expected, axis=-1)))

    @staticmethod
    def _frame_quality(image):
        grayscale = image.convert("L")
        grayscale.thumbnail((160, 90))
        statistics = ImageStat.Stat(grayscale)
        mean = float(statistics.mean[0])
        standard_deviation = float(statistics.stddev[0])
        appears_blank = standard_deviation < 1.5 and (
            mean < 3.0 or mean > 252.0
        )
        return {
            "width": int(image.width),
            "height": int(image.height),
            "luminance_mean": round(mean, 3),
            "luminance_stddev": round(standard_deviation, 3),
            "appears_blank": appears_blank,
        }

    def save_current_view(self, image=None, prefix="final"):
        if image is None:
            image = self.get_current_view()

        self.capture_directory.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S_%f"
        )

        image_path = (
            self.capture_directory /
            f"{prefix}_{timestamp}.png"
        )

        image.save(image_path, format="PNG")

        print(f"Image saved: {image_path.resolve()}")
        return str(image_path.resolve())

    def move_forward(self, magnitude=None):
        magnitude = self.move_step if magnitude is None else float(magnitude)
        yaw_rad = math.radians(self.yaw)
        new_x = self.position['x'] + magnitude * math.sin(yaw_rad)
        new_z = self.position['z'] + magnitude * math.cos(yaw_rad)

        collision, collision_type = self.check_collision(new_x, new_z)

        if collision:
            print(f"⚠️  BLOCKED! Cannot move forward. Too close to {collision_type}!")
            return False
        else:
            self.position['x'] = new_x
            self.position['z'] = new_z

            self.controller.communicate([{
                "$type": "teleport_avatar_to",
                "avatar_id": self.avatar_id,
                "position": self.position
            }])

            self._render()
            print(f"✓ Moving forward")
            return True

    def move_backward(self, magnitude=None):
        magnitude = self.move_step if magnitude is None else float(magnitude)
        yaw_rad = math.radians(self.yaw)
        new_x = self.position['x'] - magnitude * math.sin(yaw_rad)
        new_z = self.position['z'] - magnitude * math.cos(yaw_rad)

        collision, collision_type = self.check_collision(new_x, new_z)

        if collision:
            print(f"⚠️  BLOCKED! Cannot move backward. Too close to {collision_type}!")
            return False
        else:
            self.position['x'] = new_x
            self.position['z'] = new_z

            self.controller.communicate([{
                "$type": "teleport_avatar_to",
                "avatar_id": self.avatar_id,
                "position": self.position
            }])

            self._render()
            print(f"✓ Moving backward")
            return True

    def turn_body_left(self, angle=None):
        angle = self.turn_angle if angle is None else float(angle)
        self.yaw = (self.yaw - angle + 180) % 360 - 180
        yaw_rad = math.radians(self.yaw)
        forward = {
            "x": math.sin(yaw_rad),
            "y": 0,
            "z": math.cos(yaw_rad)
        }

        self.controller.communicate([{
            "$type": "set_avatar_forward",
            "avatar_id": self.avatar_id,
            "forward": forward
        }])

        self._render()
        print(f"↺ Body turning left (now facing {self.yaw}°)")
        return True

    def turn_body_right(self, angle=None):
        angle = self.turn_angle if angle is None else float(angle)
        self.yaw = (self.yaw + angle + 180) % 360 - 180
        yaw_rad = math.radians(self.yaw)
        forward = {
            "x": math.sin(yaw_rad),
            "y": 0,
            "z": math.cos(yaw_rad)
        }

        self.controller.communicate([{
            "$type": "set_avatar_forward",
            "avatar_id": self.avatar_id,
            "forward": forward
        }])

        self._render()
        print(f"↻ Body turning right (now facing {self.yaw}°)")
        return True

    def look_left(self, angle=15):
        self.camera_yaw -= angle

        self.controller.communicate([{
            "$type": "rotate_sensor_container_by",
            "avatar_id": self.avatar_id,
            "axis": "yaw",
            "angle": -angle
        }])

        self._render()
        print(f"👁 Looking left (camera yaw: {self.camera_yaw}°)")

    def look_right(self, angle=15):
        self.camera_yaw += angle

        self.controller.communicate([{
            "$type": "rotate_sensor_container_by",
            "avatar_id": self.avatar_id,
            "axis": "yaw",
            "angle": angle
        }])

        self._render()
        print(f"👁 Looking right (camera yaw: {self.camera_yaw}°)")

    def set_camera_pitch(self, pitch):
        target = min(
            self.maximum_camera_pitch,
            max(self.minimum_camera_pitch, float(pitch)),
        )
        delta = target - self.camera_pitch
        if abs(delta) < 1e-6:
            return False

        self.controller.communicate(
            [
                {
                    "$type": "rotate_sensor_container_by",
                    "avatar_id": self.avatar_id,
                    "axis": "pitch",
                    "angle": delta,
                }
            ]
        )
        self.camera_pitch = target
        self._render()
        print(f"👁 Camera pitch: {self.camera_pitch}°")
        return True

    def look_up(self, angle=None):
        angle = self.camera_pitch_step if angle is None else abs(float(angle))
        return self.set_camera_pitch(self.camera_pitch - angle)

    def look_down(self, angle=None):
        angle = self.camera_pitch_step if angle is None else abs(float(angle))
        return self.set_camera_pitch(self.camera_pitch + angle)

    def get_supported_camera_pitches(self):
        pitches = {self.initial_camera_pitch}
        value = self.minimum_camera_pitch
        while value <= self.maximum_camera_pitch + 1e-6:
            pitches.add(round(value, 6))
            value += self.camera_pitch_step
        pitches.add(self.minimum_camera_pitch)
        pitches.add(self.maximum_camera_pitch)
        return sorted(
            pitches,
            key=lambda value: (
                abs(value - self.initial_camera_pitch),
                0 if value < self.initial_camera_pitch else 1,
                value,
            ),
        )

    def list_nearby_objects(self):
        nearby = self.get_nearby_objects()

        if not nearby:
            print(f"\nNo pickupable objects nearby (within {self.interaction_distance}m)")
        else:
            print(f"\nObjects within reach ({self.interaction_distance}m):")
            for obj in nearby:
                print(f"  - {obj['name']} (ID: {obj['object_id']}) - {obj['distance']:.2f}m away")

    def _render(self):
        for i in range(3):
            self.controller.communicate([])

    def get_position(self):
        return self.position.copy()

    def get_yaw(self):
        return self.yaw

    def get_camera_angles(self):
        return {"yaw": self.camera_yaw, "pitch": self.camera_pitch}

    def get_held_object(self):
        return self.held_object

    def set_pose(self, position, yaw):
        x = float(position["x"])
        z = float(position["z"])
        y = float(position.get("y", self.position.get("y", 1.6)))
        if (
            hasattr(self.scene, "has_navigation_map")
            and self.scene.has_navigation_map()
            and not self.scene.is_navigable(x, z)
        ):
            raise ValueError(f"start pose ({x}, {z}) is not navigable.")
        self.position = {"x": x, "y": y, "z": z}
        self.yaw = (float(yaw) + 180) % 360 - 180
        yaw_rad = math.radians(self.yaw)
        self.controller.communicate([
            {"$type": "teleport_avatar_to",
             "avatar_id": self.avatar_id,
             "position": self.position},
            {"$type": "set_avatar_forward",
             "avatar_id": self.avatar_id,
             "forward": {
                 "x": math.sin(yaw_rad),
                 "y": 0,
                 "z": math.cos(yaw_rad),
             }},
        ])
        self._render()
        return self.get_state_snapshot()

    def get_state_snapshot(self):
        return {
            "position": self.get_position(),
            "yaw": self.get_yaw(),
            "camera_angles": self.get_camera_angles(),
            "held_object": (
                self.held_object.copy()
                if self.held_object is not None
                else None
            ),
            "navigation_map_enabled": bool(
                getattr(self.scene, "has_navigation_map", lambda: False)()
            ),
            "last_frame_quality": (
                dict(self.last_frame_quality)
                if self.last_frame_quality is not None
                else None
            ),
        }

    def evaluate_success(self, criteria, history=None):
        if not criteria:
            return {
                "available": False,
                "success": None,
                "reason": "No environment-side success criterion was configured.",
            }
        criterion_type = criteria.get("type")
        target_name = str(criteria.get("target_name", "")).lower()

        if criterion_type == "manual_visual_review":
            return {
                "available": False,
                "success": None,
                "independent": True,
                "requires_manual_review": True,
                "target_name": target_name or None,
                "rubric": criteria.get("rubric"),
                "reason": (
                    "The task target is architectural and has no reliable "
                    "simulator-side object state; review the saved final frame."
                ),
            }

        if criterion_type in {"all", "any"}:
            children = [
                self.evaluate_success(child, history=history)
                for child in criteria.get("criteria", [])
                if isinstance(child, dict)
            ]
            available = bool(children) and all(
                child.get("available") for child in children
            )
            values = [bool(child.get("success")) for child in children]
            success = None
            if available:
                success = all(values) if criterion_type == "all" else any(values)
            return {
                "criterion_type": criterion_type,
                "available": available,
                "success": success,
                "independent": all(
                    child.get("independent", True) for child in children
                ),
                "requires_manual_review": any(
                    child.get("requires_manual_review", False)
                    for child in children
                ),
                "reason": f"{criterion_type}_criteria={children}",
                "criteria": children,
            }

        if criterion_type == "holding":
            held_name = str((self.held_object or {}).get("name", "")).lower()
            success = bool(target_name and target_name in held_name)
            return {
                "criterion_type": criterion_type,
                "available": True,
                "success": success,
                "independent": True,
                "reason": f"held_object={held_name or None}",
            }

        if criterion_type in {"proximity", "object_proximity"}:
            target_object_id = criteria.get("target_object_id")
            if criterion_type == "object_proximity":
                try:
                    target_object_id = int(target_object_id)
                except (TypeError, ValueError):
                    return {
                        "criterion_type": criterion_type,
                        "available": False,
                        "success": None,
                        "independent": True,
                        "reason": (
                            "object_proximity requires an integer "
                            "target_object_id."
                        ),
                    }
            matches = []
            for obj in self.scene.get_object_positions():
                matches_target = (
                    obj.get("object_id") == target_object_id
                    if criterion_type == "object_proximity"
                    else target_name in str(obj.get("name", "")).lower()
                )
                if matches_target:
                    distance = math.sqrt(
                        (self.position["x"] - obj["x"]) ** 2
                        + (self.position["z"] - obj["z"]) ** 2
                    )
                    matches.append(distance)
            nearest = min(matches) if matches else None
            threshold = float(criteria.get("max_distance", 2.0))
            return {
                "criterion_type": criterion_type,
                "available": bool(matches),
                "success": nearest is not None and nearest <= threshold,
                "independent": True,
                "target_object_id": (
                    target_object_id
                    if criterion_type == "object_proximity"
                    else None
                ),
                "target_name": target_name or None,
                "distance": nearest,
                "max_distance": threshold,
                "reason": (
                    f"nearest_{target_object_id if criterion_type == 'object_proximity' else target_name}_distance={nearest}; "
                    f"threshold={threshold}"
                ),
            }

        if criterion_type == "object_visibility":
            try:
                target_object_id = int(criteria.get("target_object_id"))
                minimum_visible_pixels = int(
                    criteria.get("minimum_visible_pixels", 24)
                )
            except (TypeError, ValueError):
                return {
                    "criterion_type": criterion_type,
                    "available": True,
                    "success": False,
                    "independent": True,
                    "evaluator_error": True,
                    "reason": (
                        "object_visibility requires an integer target_object_id "
                        "and minimum_visible_pixels."
                    ),
                }
            if minimum_visible_pixels <= 0:
                return {
                    "criterion_type": criterion_type,
                    "available": True,
                    "success": False,
                    "independent": True,
                    "evaluator_error": True,
                    "target_object_id": target_object_id,
                    "reason": "minimum_visible_pixels must be positive.",
                }
            try:
                color = self._segmentation_color(target_object_id)
                if color is None:
                    raise RuntimeError(
                        "TDW returned no segmentation color for the target."
                    )
                id_pass = self.get_render_pass("_id")
                visible_pixels = self._count_instance_pixels(id_pass, color)
            except Exception as error:
                return {
                    "criterion_type": criterion_type,
                    "available": True,
                    "success": False,
                    "independent": True,
                    "evaluator_error": True,
                    "target_object_id": target_object_id,
                    "minimum_visible_pixels": minimum_visible_pixels,
                    "reason": (
                        "object_visibility evaluator failed: "
                        f"{type(error).__name__}: {error}"
                    ),
                }
            return {
                "criterion_type": criterion_type,
                "available": True,
                "success": visible_pixels >= minimum_visible_pixels,
                "independent": True,
                "target_object_id": target_object_id,
                "segmentation_color": list(color),
                "visible_pixels": visible_pixels,
                "minimum_visible_pixels": minimum_visible_pixels,
                "reason": (
                    f"target_object_id={target_object_id}; "
                    f"visible_pixels={visible_pixels}; "
                    f"minimum_visible_pixels={minimum_visible_pixels}"
                ),
            }

        if criterion_type == "minimum_successful_translations":
            try:
                minimum = int(
                    criteria.get("minimum", criteria.get("count", 1))
                )
            except (TypeError, ValueError):
                return {
                    "criterion_type": criterion_type,
                    "available": False,
                    "success": None,
                    "independent": True,
                    "reason": (
                        "minimum_successful_translations requires an integer "
                        "minimum or count."
                    ),
                }
            if minimum < 0:
                return {
                    "criterion_type": criterion_type,
                    "available": False,
                    "success": None,
                    "independent": True,
                    "reason": (
                        "minimum_successful_translations cannot be negative."
                    ),
                }
            if history is None:
                return {
                    "criterion_type": criterion_type,
                    "available": False,
                    "success": None,
                    "independent": True,
                    "reason": (
                        "No action history was provided for the translation "
                        "criterion."
                    ),
                }
            successful_translations = sum(
                item.get("action") in {"move_forward", "move_backward"}
                and item.get("outcome") == "succeeded"
                and bool(item.get("position_changed"))
                for item in history
                if isinstance(item, dict) and item.get("executed")
            )
            return {
                "criterion_type": criterion_type,
                "available": True,
                "success": successful_translations >= minimum,
                "independent": True,
                "successful_translations": successful_translations,
                "minimum": minimum,
                "reason": (
                    "successful_translations="
                    f"{successful_translations}; minimum={minimum}"
                ),
            }

        if criterion_type == "position_region":
            minimum_x = float(criteria.get("min_x", float("-inf")))
            maximum_x = float(criteria.get("max_x", float("inf")))
            minimum_z = float(criteria.get("min_z", float("-inf")))
            maximum_z = float(criteria.get("max_z", float("inf")))
            success = (
                minimum_x <= self.position["x"] <= maximum_x
                and minimum_z <= self.position["z"] <= maximum_z
            )
            return {
                "criterion_type": criterion_type,
                "available": True,
                "success": success,
                "independent": True,
                "reason": (
                    f"position=({self.position['x']}, {self.position['z']}); "
                    f"x_range=({minimum_x}, {maximum_x}); "
                    f"z_range=({minimum_z}, {maximum_z})"
                ),
            }

        return {
            "criterion_type": criterion_type,
            "available": False,
            "success": None,
            "reason": f"Unsupported success criterion: {criterion_type}",
        }

    def execute_action(self, action):
        action_to_command = {
            "move_forward": "w",
            "move_backward": "s",
            "turn_left": "a",
            "turn_right": "d",
            "look_up": "q",
            "look_down": "e",
            "pick_up": "p",
            "put_down": "o",
        }
        before = self.get_state_snapshot()
        if action not in action_to_command:
            return {
                "outcome": "invalid",
                "success": False,
                "state_before": before,
                "state_after": before,
                "position_changed": False,
                "held_object_changed": False,
            }

        result = self.execute_command(action_to_command[action])
        after = self.get_state_snapshot()
        succeeded = result is not False
        if succeeded:
            outcome = "succeeded"
        elif action in {"move_forward", "move_backward"}:
            outcome = "blocked"
        else:
            outcome = "failed"
        return {
            "outcome": outcome,
            "success": succeeded,
            "state_before": before,
            "state_after": after,
            "position_changed": before["position"] != after["position"],
            "held_object_changed": before["held_object"] != after["held_object"],
        }

    def execute_command(self, command):
        if command == "w":
            return self.move_forward()
        elif command == "s":
            return self.move_backward()
        elif command == "a":
            return self.turn_body_left()
        elif command == "d":
            return self.turn_body_right()
        elif command == "i":
            return self.save_current_view(prefix="manual")
        elif command == "q":
            return self.look_up()
        elif command == "e":
            return self.look_down()
        elif command == "p":
            return self.pick_up()
        elif command == "o":
            return self.put_down()
        elif command == "l":
            return self.list_nearby_objects()
        else:
            print("Invalid command")
            return False
