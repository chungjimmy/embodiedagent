from tdw.controller import Controller
from tdw.tdw_utils import TDWUtils
from tdw.output_data import OutputData, Rigidbodies
from heapq import heappop, heappush
import inspect
import math
import random

class Scene:
    FREE_OCCUPANCY_STATUSES = frozenset({0, 3})

    def __init__(self, controller, turn_angle_degrees=45):
        self.controller = controller
        self.turn_angle_degrees = float(turn_angle_degrees)
        if (
            not math.isfinite(self.turn_angle_degrees)
            or self.turn_angle_degrees <= 0
            or not math.isclose(
                360.0 / self.turn_angle_degrees,
                round(360.0 / self.turn_angle_degrees),
                rel_tol=0,
                abs_tol=1e-6,
            )
        ):
            raise ValueError(
                "turn_angle_degrees must be positive and divide 360 evenly."
            )
        self.object_positions = []
        self.room_size = 15
        self.using_preloaded = False
        self.occupancy_map = None
        self.occupancy_cell_size = None
        self.occupancy_raycast_y = None
        self.navigation_clearance = 0.35
        self.path_sample_spacing = 0.2
        self.occupancy_status_counts = {}

    def create_room(self, scene_name=None, seed=None):
        if seed is not None:
            random.seed(seed)

        if scene_name:
            self._load_preloaded_scene(scene_name)
        else:
            self._create_custom_room()

    def _create_custom_room(self):
        init_commands = [TDWUtils.create_empty_room(self.room_size, self.room_size)]
        self.controller.communicate(init_commands)
        self.using_preloaded = False
        print("✓ Custom room created")

    def _load_preloaded_scene(self, scene_name):
        print(f"Loading preloaded scene: {scene_name}...")

        resp = self.controller.communicate(Controller.get_add_scene(scene_name=scene_name))
        self.using_preloaded = True

        print(f"✓ Scene '{scene_name}' loaded!")

        self._catalog_preloaded_objects()

    def _catalog_preloaded_objects(self):
        from tdw.output_data import OutputData, Rigidbodies

        resp = self.controller.communicate({"$type": "send_rigidbodies", "frequency": "once"})

        for i in range(len(resp) - 1):
            r_id = OutputData.get_data_type_id(resp[i])
            if r_id == "rigi":
                rigidbodies = Rigidbodies(resp[i])
                for j in range(rigidbodies.get_num()):
                    obj_id = rigidbodies.get_id(j)
                    pos = rigidbodies.get_position(j)

                    self.object_positions.append({
                        "object_id": obj_id,
                        "name": f"scene_object_{obj_id}",
                        "x": pos[0],
                        "z": pos[2],
                        "radius": 0.3,
                        "pickupable": False
                    })

        print(f"Found {len(self.object_positions)} objects in preloaded scene")

    def add_objects(self, seed=None):
        if seed is not None:
            random.seed(seed)

        if self.using_preloaded:
            print("Using objects from preloaded scene")
            return

        objects_to_add = [
            {"name": "small_table_green_marble", "position": {"x": 0, "y": 0, "z": 0}, "radius": 0.6, "pickupable": False},
            {"name": "coffee_table_glass_round", "position": {"x": -3, "y": 0, "z": -2}, "radius": 0.7, "pickupable": False},
            {"name": "live_edge_coffee_table", "position": {"x": 2, "y": 0, "z": 2}, "radius": 0.7, "pickupable": False},
            {"name": "side_table_wood", "position": {"x": 3, "y": 0, "z": -3}, "radius": 0.4, "pickupable": False},
            {"name": "coffee_table_glass_round", "position": {"x": -5, "y": 0, "z": 3}, "radius": 0.7, "pickupable": False},

            {"name": "chair_billiani_doll", "position": {"x": -1.5, "y": 0, "z": 0}, "radius": 0.4, "pickupable": False},
            {"name": "chair_billiani_doll", "position": {"x": 1.5, "y": 0, "z": 0}, "radius": 0.4, "pickupable": False},
            {"name": "chair_eames_plastic_armchair", "position": {"x": 4, "y": 0, "z": 3}, "radius": 0.4, "pickupable": False},
            {"name": "chair_thonet_marshall", "position": {"x": 2.5, "y": 0, "z": 3.5}, "radius": 0.4, "pickupable": False},
            {"name": "chair_billiani_doll", "position": {"x": -4, "y": 0, "z": -3}, "radius": 0.4, "pickupable": False},
            {"name": "chair_eames_plastic_armchair", "position": {"x": -5.5, "y": 0, "z": 3.5}, "radius": 0.4, "pickupable": False},
            {"name": "chair_thonet_marshall", "position": {"x": 3.5, "y": 0, "z": -3.5}, "radius": 0.4, "pickupable": False},

            {"name": "cabinet_24_door_drawer_wood_beach_honey", "position": {"x": -6, "y": 0, "z": 0}, "radius": 0.6, "pickupable": False},
            {"name": "cabinet_24_door_drawer_wood_beach_honey", "position": {"x": 6, "y": 0, "z": 0}, "radius": 0.6, "pickupable": False},

            {"name": "vase_02", "position": {"x": 3, "y": 0.6, "z": -3}, "radius": 0.15, "pickupable": True},
            {"name": "vase_05", "position": {"x": -5, "y": 0.6, "z": 3}, "radius": 0.15, "pickupable": True},
            {"name": "vase_02", "position": {"x": 2, "y": 0.4, "z": 2}, "radius": 0.15, "pickupable": True},
            {"name": "vase_05", "position": {"x": -3.3, "y": 0.4, "z": -2}, "radius": 0.15, "pickupable": True},

            {"name": "iron_box", "position": {"x": 0.3, "y": 0.8, "z": 0.2}, "radius": 0.2, "pickupable": True},
            {"name": "iron_box", "position": {"x": -3, "y": 0.4, "z": -1.7}, "radius": 0.2, "pickupable": True},
            {"name": "iron_box", "position": {"x": -6, "y": 0.6, "z": 0.3}, "radius": 0.2, "pickupable": True},

            {"name": "basket_18inx18inx12iin_plastic_lattice", "position": {"x": 5, "y": 0, "z": -1}, "radius": 0.3, "pickupable": True},
            {"name": "basket_18inx18inx12iin_plastic_lattice", "position": {"x": -6.5, "y": 0, "z": 5}, "radius": 0.3, "pickupable": True},
            {"name": "basket_18inx18inx12iin_plastic_lattice", "position": {"x": 5.5, "y": 0, "z": 4}, "radius": 0.3, "pickupable": True},
        ]

        object_id = 0
        for obj in objects_to_add:
            try:
                self.controller.communicate(Controller.get_add_object(
                    model_name=obj["name"],
                    object_id=object_id,
                    position=obj["position"],
                    rotation={"x": 0, "y": random.uniform(0, 360), "z": 0}
                ))

                self.object_positions.append({
                    "object_id": object_id,
                    "name": obj["name"],
                    "x": obj["position"]["x"],
                    "z": obj["position"]["z"],
                    "radius": obj["radius"],
                    "pickupable": obj["pickupable"]
                })

                pickupable_str = "(pickupable)" if obj["pickupable"] else ""
                print(f"✓ Added {obj['name']} {pickupable_str}")
                object_id += 1
            except Exception as e:
                print(f"✗ Failed: {obj['name']} - {e}")

        print(f"\n✓ Total objects added: {len(self.object_positions)}")

        self.controller.communicate([])

    def add_configured_objects(self, objects):
        if objects in (None, []):
            return []
        if not isinstance(objects, list):
            raise ValueError("scene.task_objects must be a list when configured.")

        existing_ids = {
            item.get("object_id")
            for item in self.object_positions
            if item.get("object_id") is not None
        }
        added = []
        for index, raw_object in enumerate(objects):
            if not isinstance(raw_object, dict):
                raise ValueError(
                    f"scene.task_objects[{index}] must be an object."
                )
            model_name = str(
                raw_object.get("model_name") or raw_object.get("name") or ""
            ).strip()
            if not model_name:
                raise ValueError(
                    f"scene.task_objects[{index}].model_name is required."
                )
            label = str(raw_object.get("label") or model_name).strip()
            model_library = str(raw_object.get("library") or "").strip()
            try:
                object_id = int(raw_object["object_id"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"scene.task_objects[{index}].object_id must be an integer."
                ) from error
            if object_id in existing_ids:
                raise ValueError(
                    "scene.task_objects contains an ID already used in the "
                    f"scene: {object_id}."
                )

            position = self._normalized_object_position(
                raw_object.get("position"),
                f"scene.task_objects[{index}].position",
            )
            rotation = self._normalized_object_rotation(
                raw_object.get("rotation")
            )
            color = self._normalized_object_color(
                raw_object.get("color"),
                f"scene.task_objects[{index}].color",
            )
            scale_factor = self._normalized_scale_factor(
                raw_object.get("scale_factor", 1.0),
                f"scene.task_objects[{index}].scale_factor",
            )
            visual_description = str(
                raw_object.get("visual_description") or label
            ).strip()
            try:
                radius = float(raw_object.get("radius", 0.25))
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"scene.task_objects[{index}].radius must be numeric."
                ) from error
            if radius <= 0:
                raise ValueError(
                    f"scene.task_objects[{index}].radius must be positive."
                )

            is_static = bool(raw_object.get("static", True))
            add_object_arguments = {
                "model_name": model_name,
                "object_id": object_id,
                "position": position,
                "rotation": rotation,
            }
            if model_library:
                add_object_arguments["library"] = model_library
            add_commands = [Controller.get_add_object(**add_object_arguments)]
            if is_static:
                add_commands.append(
                    {
                        "$type": "set_kinematic_state",
                        "id": object_id,
                        "is_kinematic": True,
                        "use_gravity": False,
                    }
                )
            if any(abs(value - 1.0) >= 1e-6 for value in scale_factor.values()):
                add_commands.append(
                    {
                        "$type": "scale_object",
                        "id": object_id,
                        "scale_factor": scale_factor,
                    }
                )
            self.controller.communicate(add_commands)
            if color is not None:
                self.controller.communicate(
                    {
                        "$type": "set_color",
                        "id": object_id,
                        "color": color,
                    }
                )

            placement = raw_object.get("placement")
            placement = placement if isinstance(placement, dict) else {}
            position, grounding = self._ground_configured_object(
                object_id=object_id,
                label=label,
                position=position,
                placement=placement,
            )

            record = {
                "object_id": object_id,
                "name": label,
                "model_name": model_name,
                "x": position["x"],
                "y": position["y"],
                "z": position["z"],
                "radius": radius,
                "pickupable": bool(raw_object.get("pickupable", False)),
                "task_object": True,
                "static": is_static,
                "visual_description": visual_description,
                "scale_factor": dict(scale_factor),
            }
            if model_library:
                record["library"] = model_library
            if color is not None:
                record["color"] = dict(color)
            if placement:
                record["placement"] = dict(placement)
            if grounding is not None:
                record["grounding"] = grounding
            self.object_positions.append(record)
            existing_ids.add(object_id)
            added.append(record)
            print(
                f"✓ Added configured target {label} "
                f"(ID: {object_id}, model: {model_name})"
            )

        self.controller.communicate([])
        return added

    def _ground_configured_object(
        self,
        object_id,
        label,
        position,
        placement,
    ):
        if not bool(placement.get("ground_to_floor", False)):
            return dict(position), None
        try:
            surface_y = float(
                placement.get(
                    "resolved_surface_y",
                    placement.get("surface_y", 0.0),
                )
            )
            ground_clearance = float(placement.get("ground_clearance", 0.02))
        except (TypeError, ValueError) as error:
            raise ValueError(
                "placement.surface_y and placement.ground_clearance must be numeric."
            ) from error
        if ground_clearance < 0:
            raise ValueError("placement.ground_clearance cannot be negative.")

        bounds = self._object_bounds(object_id)
        if bounds is None:
            raise RuntimeError(
                f"TDW did not return bounds for configured target {label} "
                f"(ID {object_id}); floor grounding cannot be verified."
            )
        desired_bottom_y = surface_y + ground_clearance
        corrected = dict(position)
        corrected["y"] = float(position["y"]) + (
            desired_bottom_y - bounds["bottom_y"]
        )
        self.controller.communicate(
            {
                "$type": "teleport_object",
                "id": object_id,
                "position": corrected,
            }
        )
        verified = self._object_bounds(object_id)
        if verified is None or abs(verified["bottom_y"] - desired_bottom_y) > 0.03:
            actual = None if verified is None else verified["bottom_y"]
            raise RuntimeError(
                f"Configured target {label} (ID {object_id}) could not be "
                "grounded reliably: expected bottom y="
                f"{desired_bottom_y:.3f}, measured {actual}."
            )
        grounding = {
            "surface_y": surface_y,
            "surface_source": (
                "downward_raycast"
                if "resolved_surface_y" in placement
                else "configured_value"
            ),
            "surface_hit_object": placement.get(
                "resolved_surface_hit_object"
            ),
            "surface_object_id": placement.get(
                "resolved_surface_object_id"
            ),
            "ground_clearance": ground_clearance,
            "resolved_object_y": corrected["y"],
            "resolved_bottom_y": verified["bottom_y"],
            "resolved_top_y": verified["top_y"],
            "resolved_height": verified["top_y"] - verified["bottom_y"],
        }
        print(
            f"✓ Grounded configured target {label} (ID: {object_id}, "
            f"object y: {corrected['y']:.3f}, "
            f"bottom y: {verified['bottom_y']:.3f}, "
            f"height: {grounding['resolved_height']:.3f})"
        )
        return corrected, grounding

    def _object_bounds(self, object_id):
        from tdw.output_data import Bounds

        response = self.controller.communicate(
            {"$type": "send_bounds", "frequency": "once"}
        )
        for item in response:
            if OutputData.get_data_type_id(item) != "boun":
                continue
            bounds = Bounds(item)
            for index in range(bounds.get_num()):
                if int(bounds.get_id(index)) != int(object_id):
                    continue
                bottom = bounds.get_bottom(index)
                top = bounds.get_top(index)
                return {
                    "bottom_y": float(bottom[1]),
                    "top_y": float(top[1]),
                }
        return None

    @staticmethod
    def _normalized_object_position(value, field_name):
        if not isinstance(value, dict):
            raise ValueError(f"{field_name} must be an object with x, y, and z.")
        try:
            return {
                "x": float(value["x"]),
                "y": float(value.get("y", 0.0)),
                "z": float(value["z"]),
            }
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"{field_name} must contain numeric x, y, and z values."
            ) from error

    @staticmethod
    def _normalized_object_rotation(value):
        value = value if isinstance(value, dict) else {}
        try:
            return {
                "x": float(value.get("x", 0.0)),
                "y": float(value.get("y", 0.0)),
                "z": float(value.get("z", 0.0)),
            }
        except (TypeError, ValueError) as error:
            raise ValueError(
                "scene.task_objects rotation values must be numeric."
            ) from error

    @staticmethod
    def _normalized_object_color(value, field_name):
        if value in (None, ""):
            return None
        if not isinstance(value, dict):
            raise ValueError(
                f"{field_name} must be an object with r, g, b, and optional a."
            )
        try:
            color = {
                "r": float(value["r"]),
                "g": float(value["g"]),
                "b": float(value["b"]),
                "a": float(value.get("a", 1.0)),
            }
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"{field_name} values must be numeric between 0 and 1."
            ) from error
        if any(component < 0 or component > 1 for component in color.values()):
            raise ValueError(
                f"{field_name} values must be between 0 and 1."
            )
        return color

    @staticmethod
    def _normalized_scale_factor(value, field_name):
        if isinstance(value, (int, float)):
            scale = {
                "x": float(value),
                "y": float(value),
                "z": float(value),
            }
        elif isinstance(value, dict):
            try:
                scale = {
                    "x": float(value["x"]),
                    "y": float(value["y"]),
                    "z": float(value["z"]),
                }
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"{field_name} must contain numeric x, y, and z values."
                ) from error
        else:
            raise ValueError(
                f"{field_name} must be a positive number or Vector3 object."
            )
        if any(component <= 0 for component in scale.values()):
            raise ValueError(f"{field_name} values must be positive.")
        return scale

    def get_task_objects(self):
        return [
            dict(item)
            for item in self.object_positions
            if item.get("task_object") is True
        ]

    def resolve_task_object_positions(self, objects, initial_position):
        if not self.has_navigation_map():
            raise RuntimeError(
                "Dynamic task-object placement requires scene.navigation.enabled."
            )
        if not isinstance(objects, list) or not objects:
            return []

        resolved = []
        selected_by_id = {}
        for index, raw_object in enumerate(objects):
            if not isinstance(raw_object, dict):
                raise ValueError(
                    f"scene.task_objects[{index}] must be an object."
                )
            item = dict(raw_object)
            try:
                object_id = int(item["object_id"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"scene.task_objects[{index}].object_id must be an integer."
                ) from error
            if object_id in selected_by_id:
                raise ValueError(
                    f"scene.task_objects contains duplicate ID {object_id}."
                )

            if isinstance(item.get("position"), dict):
                item["position"] = self._normalized_object_position(
                    item["position"],
                    f"scene.task_objects[{index}].position",
                )
                item["placement"] = {
                    **(
                        item.get("placement")
                        if isinstance(item.get("placement"), dict)
                        else {}
                    ),
                    "strategy": "explicit",
                }
            else:
                placement = item.get("placement")
                if not isinstance(placement, dict):
                    raise ValueError(
                        f"scene.task_objects[{index}] needs position or placement."
                    )
                strategy = str(
                    placement.get("strategy", "farthest_reachable")
                ).strip().lower()
                if strategy not in {"farthest_reachable", "nearest_reachable"}:
                    raise ValueError(
                        "Unsupported task-object placement strategy "
                        f"{strategy!r}. Use nearest_reachable, "
                        "farthest_reachable, or position."
                    )
                anchor_x, anchor_z = self._placement_anchor(
                    placement=placement,
                    initial_position=initial_position,
                    selected_by_id=selected_by_id,
                )
                try:
                    radius = float(item.get("radius", 0.25))
                    clearance = float(
                        placement.get(
                            "clearance",
                            max(self.navigation_clearance, radius + 0.45),
                        )
                    )
                    minimum_path_distance = float(
                        placement.get("minimum_path_distance", 0.0)
                    )
                    minimum_separation = float(
                        placement.get("minimum_separation", 0.0)
                    )
                    minimum_initial_path_distance = float(
                        placement.get("minimum_initial_path_distance", 0.0)
                    )
                    minimum_departure = float(
                        placement.get("minimum_departure", 1.0)
                    )
                    minimum_view_poses = int(
                        placement.get("minimum_view_poses", 0)
                    )
                    view_minimum_distance = float(
                        placement.get("view_minimum_distance", 1.5)
                    )
                    view_maximum_distance = float(
                        placement.get("view_maximum_distance", 4.0)
                    )
                    view_preferred_distance = float(
                        placement.get("view_preferred_distance", 2.5)
                    )
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        "Task-object placement distances, view counts, and "
                        "clearance must be numeric."
                    ) from error
                if min(
                    clearance,
                    minimum_path_distance,
                    minimum_separation,
                    minimum_initial_path_distance,
                    minimum_departure,
                    minimum_view_poses,
                ) < 0:
                    raise ValueError(
                        "Task-object placement distances and clearance cannot be negative."
                    )
                if (
                    view_minimum_distance <= 0
                    or view_maximum_distance < view_minimum_distance
                ):
                    raise ValueError(
                        "Task-object placement view distances are invalid."
                    )

                candidates = self._reachable_navigation_cells(
                    anchor_x,
                    anchor_z,
                    clearance=clearance,
                )
                initial_distances = None
                if minimum_initial_path_distance > 0:
                    initial_candidates = self._reachable_navigation_cells(
                        float(initial_position["x"]),
                        float(initial_position["z"]),
                        clearance=clearance,
                    )
                    initial_distances = {
                        self._navigation_position_key(item["x"], item["z"]): item[
                            "path_distance"
                        ]
                        for item in initial_candidates
                    }
                position = self._select_task_object_candidate(
                    candidates=candidates,
                    selected=resolved,
                    minimum_path_distance=minimum_path_distance,
                    minimum_separation=minimum_separation,
                    minimum_initial_path_distance=minimum_initial_path_distance,
                    initial_distances=initial_distances,
                    minimum_departure=minimum_departure,
                    clearance=clearance,
                    farthest=strategy == "farthest_reachable",
                    minimum_view_poses=minimum_view_poses,
                    view_minimum_distance=view_minimum_distance,
                    view_maximum_distance=view_maximum_distance,
                    view_preferred_distance=view_preferred_distance,
                )
                if position is None:
                    raise ValueError(
                        "No safely reachable placement satisfies "
                        f"scene.task_objects[{index}] constraints from "
                        f"anchor=({anchor_x:g}, {anchor_z:g})."
                    )
                item["position"] = {
                    "x": position["x"],
                    "y": float(placement.get("y", item.get("y", 0.0))),
                    "z": position["z"],
                }
                item["placement"] = {
                    **placement,
                    "strategy": strategy,
                    "resolved_path_distance": round(
                        position["path_distance"], 5
                    ),
                    "resolved_anchor": {"x": anchor_x, "z": anchor_z},
                }
                if minimum_view_poses > 0:
                    item["placement"]["resolved_view_poses"] = int(
                        position["verified_view_poses"]
                    )
                if minimum_initial_path_distance > 0:
                    item["placement"]["resolved_initial_path_distance"] = round(
                        float(position["initial_path_distance"]), 5
                    )

            placement = item.get("placement")
            if (
                isinstance(placement, dict)
                and bool(placement.get("ground_to_floor", False))
            ):
                surface = self._surface_height_at(
                    item["position"]["x"],
                    item["position"]["z"],
                    origin_y=float(initial_position.get("y", 1.6)),
                )
                item["placement"] = {
                    **placement,
                    "resolved_surface_y": surface["y"],
                    "resolved_surface_hit_object": surface["hit_object"],
                    "resolved_surface_object_id": surface["object_id"],
                }

            selected_by_id[object_id] = item
            resolved.append(item)
        return resolved

    def _surface_height_at(self, x, z, origin_y=1.6):
        from tdw.output_data import Raycast

        raycast_id = self.controller.get_unique_id()
        response = self.controller.communicate(
            {
                "$type": "send_raycast",
                "origin": {
                    "x": float(x),
                    "y": float(origin_y),
                    "z": float(z),
                },
                "destination": {
                    "x": float(x),
                    "y": float(origin_y) - 100.0,
                    "z": float(z),
                },
                "id": raycast_id,
            }
        )
        for item in response[:-1]:
            if OutputData.get_data_type_id(item) != "rayc":
                continue
            raycast = Raycast(item)
            if raycast.get_raycast_id() != raycast_id or not raycast.get_hit():
                continue
            point = raycast.get_point()
            hit_object = bool(raycast.get_hit_object())
            return {
                "y": float(point[1]),
                "hit_object": hit_object,
                "object_id": (
                    int(raycast.get_object_id()) if hit_object else None
                ),
            }
        raise RuntimeError(
            "No support surface was found beneath configured target position "
            f"({float(x):.3f}, {float(z):.3f})."
        )

    @staticmethod
    def _navigation_position_key(x, z):
        return round(float(x), 6), round(float(z), 6)

    @staticmethod
    def _placement_anchor(placement, initial_position, selected_by_id):
        reference_id = placement.get("reference_object_id")
        anchor = placement.get("anchor")
        if anchor not in (None, ""):
            if reference_id not in (None, ""):
                raise ValueError(
                    "placement.anchor and placement.reference_object_id "
                    "cannot both be set."
                )
            try:
                return float(anchor["x"]), float(anchor["z"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    "placement.anchor must be an object with numeric x and z."
                ) from error
        if reference_id in (None, ""):
            return float(initial_position["x"]), float(initial_position["z"])
        try:
            reference_id = int(reference_id)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "placement.reference_object_id must be an integer."
            ) from error
        reference = selected_by_id.get(reference_id)
        if reference is None:
            raise ValueError(
                "placement.reference_object_id must refer to an earlier "
                f"task object; {reference_id} has not been resolved."
            )
        position = reference["position"]
        return float(position["x"]), float(position["z"])

    def _reachable_navigation_cells(self, start_x, start_z, clearance):
        if not self.has_navigation_map():
            return []
        import numpy as np

        positions = np.asarray(self.occupancy_map.positions, dtype=float)
        occupancy = np.asarray(self.occupancy_map.occupancy_map)
        traversable = np.zeros(occupancy.shape, dtype=bool)
        for index in np.ndindex(occupancy.shape):
            if occupancy[index] not in self.FREE_OCCUPANCY_STATUSES:
                continue
            x, z = positions[index][0], positions[index][1]
            traversable[index] = bool(
                self.is_navigable(x, z, clearance=clearance)
            )
        available = np.argwhere(traversable)
        if not len(available):
            return []

        start = np.asarray([float(start_x), float(start_z)])
        candidate_positions = positions[traversable]
        nearest = int(
            np.argmin(np.linalg.norm(candidate_positions - start, axis=1))
        )
        start_index = tuple(int(value) for value in available[nearest])
        distances = {start_index: 0.0}
        queue = [(0.0, start_index)]
        shape = occupancy.shape
        while queue:
            distance, current = heappop(queue)
            if distance != distances.get(current):
                continue
            for x_delta in (-1, 0, 1):
                for z_delta in (-1, 0, 1):
                    if x_delta == 0 and z_delta == 0:
                        continue
                    neighbor = (current[0] + x_delta, current[1] + z_delta)
                    if not (
                        0 <= neighbor[0] < shape[0]
                        and 0 <= neighbor[1] < shape[1]
                        and traversable[neighbor]
                    ):
                        continue
                    step_distance = float(
                        np.linalg.norm(
                            positions[neighbor] - positions[current]
                        )
                    )
                    proposed = distance + step_distance
                    if proposed >= distances.get(neighbor, float("inf")):
                        continue
                    distances[neighbor] = proposed
                    heappush(queue, (proposed, neighbor))
        return [
            {
                "x": float(positions[index][0]),
                "z": float(positions[index][1]),
                "path_distance": float(distance),
            }
            for index, distance in distances.items()
        ]

    def _select_task_object_candidate(
        self,
        candidates,
        selected,
        minimum_path_distance,
        minimum_separation,
        minimum_initial_path_distance,
        initial_distances,
        minimum_departure,
        clearance,
        farthest,
        minimum_view_poses=0,
        view_minimum_distance=1.5,
        view_maximum_distance=4.0,
        view_preferred_distance=2.5,
    ):
        ordered = sorted(
            candidates,
            key=lambda item: (item["path_distance"], item["x"], item["z"]),
            reverse=bool(farthest),
        )
        for candidate in ordered:
            if candidate["path_distance"] < minimum_path_distance:
                continue
            if minimum_initial_path_distance > 0:
                initial_path_distance = (initial_distances or {}).get(
                    self._navigation_position_key(candidate["x"], candidate["z"])
                )
                if (
                    initial_path_distance is None
                    or initial_path_distance < minimum_initial_path_distance
                ):
                    continue
                candidate = {
                    **candidate,
                    "initial_path_distance": float(initial_path_distance),
                }
            if any(
                math.hypot(
                    candidate["x"] - float(item["position"]["x"]),
                    candidate["z"] - float(item["position"]["z"]),
                ) < minimum_separation
                for item in selected
            ):
                continue
            if minimum_departure > 0 and not self.has_safe_departure(
                candidate["x"],
                candidate["z"],
                distance=minimum_departure,
                clearance=clearance,
            ):
                continue
            if minimum_view_poses > 0:
                view_poses = self.target_view_candidates(
                    candidate["x"],
                    candidate["z"],
                    minimum_distance=view_minimum_distance,
                    maximum_distance=view_maximum_distance,
                    preferred_distance=view_preferred_distance,
                    clearance=self.navigation_clearance,
                    max_candidates=minimum_view_poses,
                )
                if len(view_poses) < minimum_view_poses:
                    continue
                candidate = {
                    **candidate,
                    "verified_view_poses": len(view_poses),
                }
            return candidate
        return None

    def update_object_position(self, object_id, new_x, new_z):
        for obj in self.object_positions:
            if obj["object_id"] == object_id:
                obj["x"] = new_x
                obj["z"] = new_z
                print(f"Updated position for object {object_id} to ({new_x:.2f}, {new_z:.2f})")
                break

    def get_object_positions(self):
        return self.object_positions

    def get_room_size(self):
        return self.room_size

    def setup_navigation(
        self,
        cell_size=0.25,
        raycast_y=1.6,
        clearance=0.35,
        path_sample_spacing=0.2,
    ):
        try:
            from tdw.add_ons.occupancy_map import OccupancyMap
        except ImportError as error:
            raise RuntimeError(
                "This TDW installation does not provide OccupancyMap. Update "
                "TDW or disable scene.navigation.enabled explicitly."
            ) from error

        prior_map = self.occupancy_map
        if prior_map is not None:
            try:
                self.controller.add_ons.remove(prior_map)
            except (AttributeError, ValueError):
                pass
        self.occupancy_map = None

        requested_cell_size = float(cell_size)
        requested_raycast_y = float(raycast_y)
        constructor_parameters = inspect.signature(OccupancyMap).parameters
        constructor_options = {}
        if "cell_size" in constructor_parameters:
            constructor_options["cell_size"] = requested_cell_size
        occupancy_map = OccupancyMap(**constructor_options)
        self.controller.add_ons.append(occupancy_map)
        self.controller.communicate([])
        generate_parameters = inspect.signature(
            occupancy_map.generate
        ).parameters
        generate_options = {}
        if "cell_size" in generate_parameters:
            generate_options["cell_size"] = requested_cell_size
        if "raycast_y" in generate_parameters:
            generate_options["raycast_y"] = requested_raycast_y
        occupancy_map.generate(**generate_options)
        self.controller.communicate([])

        occupancy = getattr(occupancy_map, "occupancy_map", None)
        positions = getattr(occupancy_map, "positions", None)
        if occupancy is None or positions is None:
            raise RuntimeError("TDW returned no occupancy-map data.")
        if getattr(occupancy, "size", 0) == 0 or getattr(positions, "size", 0) == 0:
            raise RuntimeError("TDW returned an empty occupancy map.")

        import numpy as np

        self.occupancy_map = occupancy_map
        self.occupancy_cell_size = self._infer_occupancy_cell_size(
            positions,
            fallback=requested_cell_size,
        )
        self.occupancy_raycast_y = (
            requested_raycast_y
            if "raycast_y" in generate_parameters
            else None
        )
        self.navigation_clearance = max(0.0, float(clearance))
        self.path_sample_spacing = max(0.05, float(path_sample_spacing))
        shape = getattr(occupancy, "shape", None)
        statuses, counts = np.unique(np.asarray(occupancy), return_counts=True)
        self.occupancy_status_counts = {
            int(status): int(count)
            for status, count in zip(statuses, counts)
        }
        free_count = sum(
            self.occupancy_status_counts.get(status, 0)
            for status in self.FREE_OCCUPANCY_STATUSES
        )
        print(
            "✓ Navigation occupancy map generated: "
            f"shape={shape}, cell_size={self.occupancy_cell_size:g}, "
            f"raycast_y={self.occupancy_raycast_y}, "
            f"statuses={self.occupancy_status_counts}, "
            f"traversable_cells={free_count}"
        )
        if free_count == 0:
            raise RuntimeError(
                "TDW generated an occupancy map with no traversable cells. "
                f"status_counts={self.occupancy_status_counts}, "
                f"raycast_y={self.occupancy_raycast_y}. "
                "For a closed indoor scene, set scene.navigation.raycast_y "
                "to a height below the ceiling and above the floor."
            )

    @staticmethod
    def _infer_occupancy_cell_size(positions, fallback):
        import numpy as np

        positions = np.asarray(positions, dtype=float)
        spacings = []
        for axis in (0, 1):
            values = np.unique(positions[..., axis])
            differences = np.diff(np.sort(values))
            positive = differences[differences > 1e-6]
            if positive.size:
                spacings.append(float(positive.min()))
        return min(spacings) if spacings else float(fallback)

    def has_navigation_map(self):
        return self.occupancy_map is not None

    def _flat_navigation_arrays(self):
        if not self.has_navigation_map():
            return None, None
        import numpy as np

        positions = np.asarray(self.occupancy_map.positions, dtype=float)
        occupancy = np.asarray(self.occupancy_map.occupancy_map)
        if positions.ndim < 2 or positions.shape[-1] < 2:
            raise RuntimeError(
                f"Unexpected occupancy positions shape: {positions.shape}"
            )
        flat_positions = positions[..., :2].reshape(-1, 2)
        flat_occupancy = occupancy.reshape(-1)
        if len(flat_positions) != len(flat_occupancy):
            raise RuntimeError(
                "Occupancy positions and status grids have different sizes."
            )
        return flat_positions, flat_occupancy

    def is_navigable(self, x, z, clearance=None):
        if not self.has_navigation_map():
            return None
        import numpy as np

        positions, occupancy = self._flat_navigation_arrays()
        point = np.asarray([float(x), float(z)])
        distances = np.linalg.norm(positions - point, axis=1)
        nearest_distance = float(distances.min())
        maximum_cell_distance = max(
            self.occupancy_cell_size or 0.25,
            0.05,
        )
        if nearest_distance > maximum_cell_distance:
            return False

        requested_clearance = (
            self.navigation_clearance if clearance is None else float(clearance)
        )
        if requested_clearance <= 0:
            nearby = np.zeros(len(distances), dtype=bool)
            nearby[distances.argmin()] = True
        else:
            nearby = distances <= requested_clearance
            if not nearby.any():
                nearby[distances.argmin()] = True
        return bool(
            np.all(np.isin(occupancy[nearby], tuple(self.FREE_OCCUPANCY_STATUSES)))
        )

    def is_path_navigable(
        self, start_x, start_z, end_x, end_z, clearance=None
    ):
        if not self.has_navigation_map():
            return None
        distance = math.hypot(end_x - start_x, end_z - start_z)
        samples = max(1, int(math.ceil(distance / self.path_sample_spacing)))
        for index in range(1, samples + 1):
            fraction = index / samples
            x = start_x + (end_x - start_x) * fraction
            z = start_z + (end_z - start_z) * fraction
            if not self.is_navigable(x, z, clearance=clearance):
                return False
        return True

    def has_safe_departure(self, x, z, distance, clearance=None):
        if not self.has_navigation_map():
            return None
        requested_clearance = (
            self.navigation_clearance if clearance is None else float(clearance)
        )
        if not self.is_navigable(x, z, clearance=requested_clearance):
            return False
        departure_distance = max(0.0, float(distance))
        if departure_distance == 0:
            return True
        heading_count = int(round(360.0 / self.turn_angle_degrees))
        for heading_index in range(heading_count):
            heading = heading_index * self.turn_angle_degrees
            radians = math.radians(heading)
            end_x = float(x) + departure_distance * math.sin(radians)
            end_z = float(z) + departure_distance * math.cos(radians)
            if self.is_path_navigable(
                x,
                z,
                end_x,
                end_z,
                clearance=requested_clearance,
            ):
                return True
        return False

    def nearest_navigable_position(
        self, x, z, clearance=None, min_departure=0.0
    ):
        if not self.has_navigation_map():
            return None
        import numpy as np

        positions, occupancy = self._flat_navigation_arrays()
        free_positions = positions[
            np.isin(occupancy, tuple(self.FREE_OCCUPANCY_STATUSES))
        ]
        if len(free_positions) == 0:
            return None
        point = np.asarray([float(x), float(z)])
        candidate_indices = np.argsort(
            np.linalg.norm(free_positions - point, axis=1)
        )
        for index in candidate_indices:
            candidate = free_positions[index]
            candidate_x = float(candidate[0])
            candidate_z = float(candidate[1])
            if not self.is_navigable(
                candidate_x,
                candidate_z,
                clearance=clearance,
            ):
                continue
            if (
                float(min_departure) > 0
                and not self.has_safe_departure(
                    candidate_x,
                    candidate_z,
                    distance=min_departure,
                    clearance=clearance,
                )
            ):
                continue
            return {"x": candidate_x, "z": candidate_z}
        return None

    def target_view_candidates(
        self,
        target_x,
        target_z,
        minimum_distance=2.0,
        maximum_distance=3.5,
        preferred_distance=2.5,
        clearance=None,
        max_candidates=24,
    ):
        if not self.has_navigation_map():
            return []
        try:
            minimum_distance = float(minimum_distance)
            maximum_distance = float(maximum_distance)
            preferred_distance = float(preferred_distance)
            max_candidates = int(max_candidates)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "Target-visibility preflight distances must be numeric."
            ) from error
        if (
            minimum_distance <= 0
            or maximum_distance < minimum_distance
            or max_candidates <= 0
        ):
            raise ValueError(
                "Target-visibility preflight distances are invalid."
            )

        import numpy as np

        positions, occupancy = self._flat_navigation_arrays()
        candidates = []
        for position, status in zip(positions, occupancy):
            if status not in self.FREE_OCCUPANCY_STATUSES:
                continue
            candidate_x = float(position[0])
            candidate_z = float(position[1])
            distance = math.hypot(
                float(target_x) - candidate_x,
                float(target_z) - candidate_z,
            )
            if not minimum_distance <= distance <= maximum_distance:
                continue
            if not self.is_navigable(
                candidate_x,
                candidate_z,
                clearance=clearance,
            ):
                continue
            if not self.is_path_navigable(
                candidate_x,
                candidate_z,
                float(target_x),
                float(target_z),
                clearance=clearance,
            ):
                continue
            bearing = math.degrees(
                math.atan2(
                    float(target_x) - candidate_x,
                    float(target_z) - candidate_z,
                )
            )
            yaw = (
                round(bearing / self.turn_angle_degrees)
                * self.turn_angle_degrees
            ) % 360.0
            angular_error = abs(((bearing - yaw + 180.0) % 360.0) - 180.0)
            candidates.append(
                {
                    "x": candidate_x,
                    "z": candidate_z,
                    "yaw": yaw,
                    "distance": distance,
                    "angular_error": angular_error,
                }
            )
        candidates.sort(
            key=lambda item: (
                abs(item["distance"] - preferred_distance),
                item["angular_error"],
                item["x"],
                item["z"],
            )
        )
        return candidates[:max_candidates]

    def navigation_diagnostics(self):
        return {
            "cell_size": self.occupancy_cell_size,
            "raycast_y": self.occupancy_raycast_y,
            "status_counts": dict(self.occupancy_status_counts),
            "traversable_statuses": sorted(self.FREE_OCCUPANCY_STATUSES),
        }
