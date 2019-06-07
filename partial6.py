from big_ol_pile_of_manim_imports import *

# TODO, think about the use of kwargs.
# Better to have a pack of all default configs.
# How to make the line hided in the bottom when creating?
# The station_name has the same question. How does the layer work?
# Add BackGroundRectangle to the station_name.
# Set show_name(boolean) as an option.

# tex_mobject.py line 60, use reduce for piles of and & or
# Rename AnimateStation, PersonalizedCoordinateSystem
# Sort the classes


# The rate between unit and stroke_width is simply 100:1.
DEFAULT_UNIT_AND_STROKE_RATIO = 100
DEFAULT_UNIT = 0.08
# This should be 1, but as an experimantal project, I put 0.625 here temporary
#DEFAULT_ROUTE_WIDTH_RATIO = 1
DEFAULT_ROUTE_WIDTH_RATIO = 0.625
# The ratio of the radius and the unit
DEFAULT_ROUTE_RADIUS_RATIO = 2
# Units per second
DEFAULT_PLAY_SPEED = 15.0
DEFAULT_SHOW_RUN_TIME = 1.0

DEFAULT_POINT_RADIUS_RATIO = 0.8
DEFAULT_NEW_STATION_RADIUS_RATIO = 0.9
DEFAULT_INTERCHANGE_STATION_RADIUS_RATIO = 1.2
DEFAULT_NEW_STATION_STROKE_RATIO = 0.3
DEFAULT_INTERCHANGE_STATION_STROKE_RATIO = 0.1
DEFAULT_ROUTE_OPACITY = 0.6######
DEFAULT_STATION_OPACITY = 1.0

#DEFAULT_OUTER_RING_WIDTH_FACTOR = 1.25
#DEFAULT_INNER_RING_WIDTH_FACTOR = 0.75
DEFAULT_NEW_STATION_FADE_FACTOR = 3.0
DEFAULT_POINT_FADE_FACTOR = 4.0
#DEFAULT_NEW_STATION_STROKE_WIDTH = 3
#DEFAULT_INTERCHANGE_STATION_STROKE_WIDTH = 1
DEFAULT_NAME_SCALE_FACTOR = 0.2

DEFAULT_NEW_STATION_ANIMATION = lambda obj, t, a, b: FadeInFromLarge(
    obj,
    scale_factor=DEFAULT_NEW_STATION_FADE_FACTOR,
    run_time=t,
    rate_func=squish_rate_func(smooth, a / t, b / t)
)
DEFAULT_POINT_ANIMATION = lambda obj, t, a, b: FadeInFromLarge(
    obj,
    scale_factor=DEFAULT_POINT_FADE_FACTOR,
    run_time=t,
    rate_func=squish_rate_func(smooth, a / t, b / t)
)
DEFAULT_GROW_STATION_ANIMATION = lambda objs, t, a, b: ReplacementTransform(
    *objs,
    run_time=t,
    rate_func=squish_rate_func(smooth, a / t, b / t)
)
DEFAULT_SUB_POINT_ANIMATION = lambda obj, t, a, b: GrowFromCenter(
    obj,
    run_time=t,
    rate_func=squish_rate_func(linear, a / t, b / t)
)
DEFAULT_NEW_STATION_NAME_ANIMATION = lambda obj, t, a, b: Write(
    obj,
    run_time=t,
    rate_func=squish_rate_func(linear, a / t, b / t)
)
DEFAULT_INTERCHANGE_STATION_NAME_ANIMATION = lambda objs, t, a, b: ReplacementTransform(
    *objs,
    run_time=t,
    rate_func=squish_rate_func(smooth, a / t, b / t)
)


def np_int(*elements):
    return np.array(elements, dtype="int64")


PLAIN_ORIGIN = np_int(0, 0)
PLAIN_RIGHT = np_int(1, 0)
PLAIN_LEFT = np_int(-1, 0)
PLAIN_UP = np_int(0, 1)
PLAIN_DOWN = np_int(0, -1)


# Global the data.
# dict_key: the to-extend coordinate of the station
# dict_value: a complete station as a VGroup
global_station_data = {}
global_reversed_data = {}


# The parameter required_shape should be a tuple which consists positive integers or None.
# e.g. required_shape=(3, 1, None)
def is_np_data(data, required_dtype=None, required_shape=None):
    if type(data) != np.ndarray:
        return False
    if required_dtype is None:
        if "int" not in str(data.dtype) and "float" not in str(data.dtype):
            return False
    elif required_dtype != str(data.dtype):
        return False
    if required_shape is None:
        return True
    data_shape = np.shape(data)
    required_dimension = len(required_shape)
    if len(data_shape) != required_dimension:
        return False
    for k in range(required_dimension):
        if required_shape[k] is not None and data_shape[k] != required_shape[k]:
            return False
    return True


def round_int(data, dtype=np.int64):
    assert is_np_data(data)
    data = np.rint(data).astype(dtype)
    return data


def get_cases_linear_func(key_points, a=0, b=1, partial_data=True, add_origin=True, complete_rate=True, early_jump=True):
    assert is_np_data(key_points, required_shape=(None, 2))
    if partial_data:
        key_points = np.array([
            np.sum(key_points[:k+1], axis=0)
            for k in range(len(key_points))
        ])
    if add_origin:
        key_points = np.r_[[PLAIN_ORIGIN], key_points]
    if complete_rate:
        # Make the last key_point sit on (1, 1).
        last_point_matrix = np.diag(1 / key_points[-1])
        key_points = np.dot(key_points, last_point_matrix)
    # Shrink the abscissa from [0, 1] to [a, b].
    x, y = key_points.T
    key_points = np.c_[(b - a) * x + a, y]

    def result(t):
        for k in range(len(key_points) - 1):
            x0, y0 = key_points[k]
            x1, y1 = key_points[k+1]
            assert x0 <= x1
            if t >= x0 and t <= x1:
                try:
                    return ((y1 - y0) * t + x1 * y0 - x0 * y1) / (x1 - x0)
                except ZeroDivisionError:
                    return y1 if early_jump else y0
        # Avoid error caused by float comparisons.
        if t < key_points[0][0]:
            return key_points[0][1]
        elif t > key_points[-1][0]:
            return key_points[-1][1]
    return result


class PersonalizedCoordinateSystem(object):
    CONFIG = {
        "unit": DEFAULT_UNIT,
    }

    def __init__(self, **kwargs):
        digest_config(self, kwargs)

    def change_unit(self, length, unit_length=None):
        unit_length = unit_length or self.unit
        return length * unit_length

    def recover_unit(self, length, unit_length=None):
        unit_length = unit_length or self.unit
        return length / unit_length

    # The origin is in the bottom left corner of the screen.
    def change_coordinate(self, two_d_coordinate, unit_length=None):
        unit_length = unit_length or self.unit
        x = unit_length * two_d_coordinate[0] - FRAME_X_RADIUS
        y = unit_length * two_d_coordinate[1] - FRAME_Y_RADIUS
        return np.array([x, y, 0])

    def recover_coordinate(self, three_d_coordinate, unit_length=None):
        unit_length = unit_length or self.unit
        x = (three_d_coordinate[0] + FRAME_X_RADIUS) / unit_length
        y = (three_d_coordinate[1] + FRAME_Y_RADIUS) / unit_length
        return np.array([x, y])


class Route(VMobject, PersonalizedCoordinateSystem):
    CONFIG = {
        "route_width_ratio": DEFAULT_ROUTE_WIDTH_RATIO,
        "route_radius_ratio": DEFAULT_ROUTE_RADIUS_RATIO,
        "route_color": WHITE,
        "route_opacity": DEFAULT_ROUTE_OPACITY,
        "background_color": None,
        "background_opacity": None,
        "special_radius_ratios": None,
        "loop": False,
        "play_speed": DEFAULT_PLAY_SPEED,
        "check_integer": True,
        "unit_and_stroke_ratio": DEFAULT_UNIT_AND_STROKE_RATIO,
    }

    def __init__(self, orderd_coordinates, **kwargs):
        VMobject.__init__(self, **kwargs)
        PersonalizedCoordinateSystem.__init__(self, **kwargs)
        if self.check_integer:
            assert is_np_data(orderd_coordinates, required_dtype="int64", required_shape=(None, 2))
        else:
            assert is_np_data(orderd_coordinates, required_shape=(None, 2))
        self.orderd_coordinates = orderd_coordinates
        vertices = np.array([
            self.change_coordinate(coordinate)
            for coordinate in orderd_coordinates
        ])
        self.num_of_lines = np.shape(orderd_coordinates)[0]
        if self.loop:
            self.set_points_as_corners([*vertices, vertices[0]])
        else:
            self.set_points_as_corners([*vertices, vertices[-1]])
            self.num_of_lines -= 1
        assert self.num_of_lines > 0
        self.vertices = self.get_start_anchors()
        self.set_component_as_attr()
        self.round_corners()
        self.set_style(
            stroke_width=self.change_unit(self.route_width_ratio * self.unit_and_stroke_ratio),
            stroke_color=self.route_color,
            stroke_opacity=self.route_opacity,
            fill_color=self.background_color,
            fill_opacity=self.background_opacity
        )
        self.total_time = self.recover_unit(self.total_length) / self.play_speed

    def get_radius_list(self):
        num_of_arcs = self.num_of_lines
        if not self.loop:
            num_of_arcs -= 1
        route_radius_ratios = np.ones(num_of_arcs, dtype="int64") * self.route_radius_ratio
        if self.special_radius_ratios is not None:
            assert is_np_data(self.special_radius_ratios, required_shape=(None, 2))
            for k in self.special_radius_ratios:
                assert k[1] >= 0
                route_radius_ratios[k[0]] = k[1]
        return route_radius_ratios

    def get_arc_list(self):
        vertices = self.vertices
        arcs = []
        adjacent_vertices = list(adjacent_n_tuples(vertices, 3))
        if not self.loop:
            adjacent_vertices = adjacent_vertices[:-2]
        radius_list = self.get_radius_list()
        for k, v in enumerate(adjacent_vertices):
            v1, v2, v3 = v
            radius = self.change_unit(radius_list[k])
            vect1 = v2 - v1
            vect2 = v3 - v2
            unit_vect1 = normalize(vect1)
            unit_vect2 = normalize(vect2)
            angle = angle_between_vectors(vect1, vect2)
            # Distance between vertex and start of the arc
            cut_off_length = radius * np.tan(angle / 2)
            # Determines counterclockwise vs. clockwise
            angle_sign = np.sign(np.cross(vect1, vect2)[2])
            arc = ArcBetweenPoints(
                v2 - unit_vect1 * cut_off_length,
                v2 + unit_vect2 * cut_off_length,
                angle=angle_sign * angle
            )
            arc.radius = radius
            arc.__setattr__("length", arc.radius * abs(arc.angle))
            arcs.append(arc)
        return arcs

    def get_line_list(self):
        return [line for line in self.get_component_list() if isinstance(line, Line)]

    def set_component_as_attr(self):
        vertices = self.vertices
        arcs = self.get_arc_list()
        components = []
        adjacent_arcs = list(adjacent_pairs(arcs))
        if not self.loop:
            first_line = Line(vertices[0], arcs[0].get_start())
            components.extend([first_line])
            adjacent_arcs = adjacent_arcs[:-1]
            for arc1, arc2 in adjacent_arcs:
                line = Line(arc1.get_end(), arc2.get_start())
                components.extend([arc1, line])
            last_arc = arcs[-1]
            last_line = Line(last_arc.get_end(), vertices[-1])
            components.extend([last_arc, last_line])
        else:
            for arc1, arc2 in adjacent_arcs:
                line = Line(arc1.get_end(), arc2.get_start())
                components.extend([arc1, line])
        for line in components:
            if isinstance(line, Line):
                line.__setattr__("length", get_norm(line.get_vector()))
        self.components = components
        return self

    def get_component_list(self):
        return self.components

    def round_corners(self):
        components = self.get_component_list()
        self.cases_key_points = np.zeros((len(components), 2), dtype="float64")
        self.clear_points()
        for k, component in enumerate(components):
            self.append_points(component.points)
            n = 1 if isinstance(component, Line) else (component.num_components - 1)
            self.cases_key_points[k] = np.array([component.length, n])
        self.total_length = np.sum(self.cases_key_points, axis=0)[0]
        return self

    def get_route_linear_func(self, a, b):
        def result(t):
            return get_cases_linear_func(self.cases_key_points, a=a, b=b)(t)
        return result

    def in_range(self, value, min_val, max_val, edge_involved=True):
        val = (value - min_val) * (value - max_val)
        return val <= 0 if edge_involved else val < 0

    def simple_proportion(self, value, min_val, max_val):
        if min_val == max_val:
            return np.nan
        return (value - min_val) / (max_val - min_val)

    def choose_proportion(self, point_coordinate, start_point, end_point):
        choices = [
            self.simple_proportion(*point_val)
            for point_val in zip(point_coordinate, start_point, end_point)
        ]
        result = choices[0] if choices[0] is not np.nan else choices[1]
        return result

    def point_on_line_proportion(self, point_coordinate, start_point, end_point):
        assert all([
            is_np_data(data, required_shape=(2,))
            for data in (point_coordinate, start_point, end_point)
        ])
        x, y = point_coordinate
        x0, y0 = start_point
        x1, y1 = end_point
        # Float comparisons are really tricky...
        if not all([self.in_range(x, x0, x1), self.in_range(y, y0, y1)]):
            return
        # 3d determinant
        if (y1 - y0) * x + (x0 - x1) * y - x0 * y1 + x1 * y0 != 0:
            return
        result = self.choose_proportion(point_coordinate, start_point, end_point)
        return result

    def point_on_route_proportion(self, point_coordinate):
        assert is_np_data(point_coordinate, required_shape=(2,))
        assert not self.loop
        lines = self.get_line_list()
        components = self.get_component_list()
        for k in range(self.num_of_lines):
            if self.point_on_line_proportion(
                point_coordinate,
                self.orderd_coordinates[k],
                self.orderd_coordinates[k+1]
            ) is not None:
                line_start = self.recover_coordinate(lines[k].get_start())
                line_end = self.recover_coordinate(lines[k].get_end())
                local_proportion = self.choose_proportion(point_coordinate, line_start, line_end)
                if self.in_range(local_proportion, 0, 1):
                    index = k
                    break
        else:
            return
        partial_length = sum([
            component.length
            for component in components[:2 * index]
        ]) + local_proportion * lines[index].length
        return partial_length / self.total_length


class StationFrame(Route):
    CONFIG = {
        "horizontal": True,
        "station_size": None,
        "frame_stroke_ratio": None,
        "frame_color": None,
        "frame_opacity": None,
        "frame_background_color": None,
        "frame_background_opacity": None,
        "radius_ratio": None,
    }

    def __init__(self, center_coordinate, **kwargs):
        digest_config(self, kwargs)
        self.center_coordinate = center_coordinate
        orderd_coordinates = self.get_ordered_coordinates()
        Route.__init__(
            self,
            orderd_coordinates,
            check_integer=False,
            loop=True,
            route_color=self.frame_color,
            route_width_ratio=self.frame_stroke_ratio,
            route_radius_ratio=self.radius_ratio / 2,
            route_opacity=self.frame_opacity,
            background_color=self.frame_background_color,
            background_opacity=self.frame_background_opacity
        )

    def get_ordered_coordinates(self):
        center = self.center_coordinate
        x = (self.station_size - 1 + self.radius_ratio) / 2
        y = self.radius_ratio / 2
        if not self.horizontal:
            x, y = y, x
        right = center + x * PLAIN_RIGHT
        ur = center + x * PLAIN_RIGHT + y * PLAIN_UP
        ul = center + x * PLAIN_LEFT + y * PLAIN_UP
        dl = center + x * PLAIN_LEFT + y * PLAIN_DOWN
        dr = center + x * PLAIN_RIGHT + y * PLAIN_DOWN
        coordinates = np.c_[right, ur, ul, dl, dr].T
        return coordinates


class StationPoint(Dot, PersonalizedCoordinateSystem):
    CONFIG = {
        "point_radius_ratio": None,
        "point_color": None,
        "point_opacity": None,
    }

    def __init__(self, point_coordinate, **kwargs):
        PersonalizedCoordinateSystem.__init__(self, **kwargs)
        Dot.__init__(
            self,
            point=self.change_coordinate(point_coordinate),
            radius=self.change_unit(self.point_radius_ratio / 2),
            fill_color=self.point_color,
            fill_opacity=self.point_opacity,
            stroke_width=0
        )


class Station(VGroup):
    CONFIG = {
        "station_name": None,
        "horizontal": True,
        "reduce_points": 0,
        "point_radius_ratio": DEFAULT_POINT_RADIUS_RATIO,
        "new_station_config": {
            "radius_ratio": DEFAULT_NEW_STATION_RADIUS_RATIO,
            "frame_stroke_ratio": DEFAULT_NEW_STATION_STROKE_RATIO,
        },
        "interchange_station_config": {
            "radius_ratio": DEFAULT_INTERCHANGE_STATION_RADIUS_RATIO,
            "frame_stroke_ratio": DEFAULT_INTERCHANGE_STATION_STROKE_RATIO,
        },
        "station_opacity": DEFAULT_STATION_OPACITY,
        "frame_color": WHITE,
        "frame_background_color": BLACK,
    }

    # About kwargs...
    def __init__(self, coordinate_group, color_group, **kwargs):
        digest_config(self, kwargs)
        assert is_np_data(coordinate_group, required_dtype="int64", required_shape=(None, 2))
        assert np.shape(coordinate_group)[0] == len(color_group)
        self.coordinate_group = coordinate_group
        self.color_group = color_group
        self.kwargs = kwargs
        self.station_size = len(color_group)
        self.center_coordinate = np.average(coordinate_group, axis=0)
        if self.station_size == 1:
            self.reduce_points = 1
        station_frame = self.get_station_frame()
        self.station_points = self.get_station_points()
        VGroup.__init__(self, station_frame, *self.station_points)

    def get_station_frame(self):
        if self.station_size == 1:
            self.frame_color = self.color_group[0]
        config_data = self.new_station_config if self.station_size == 1 else self.interchange_station_config
        radius_ratio = config_data["radius_ratio"]
        frame_stroke_ratio = config_data["frame_stroke_ratio"]
        station_frame = StationFrame(
            self.center_coordinate,
            horizontal=self.horizontal,
            station_size=self.station_size,
            radius_ratio=radius_ratio,
            frame_stroke_ratio=frame_stroke_ratio,
            frame_color=self.frame_color,
            frame_opacity=self.station_opacity,
            frame_background_color=self.frame_background_color,
            frame_background_opacity=self.station_opacity
        )
        return station_frame

    def get_station_points(self):
        station_points = []
        for point_coordinate, point_color in zip(self.coordinate_group, self.color_group):
            station_point = StationPoint(
                point_coordinate,
                point_radius_ratio=self.point_radius_ratio,
                point_color=point_color,
                point_opacity=self.station_opacity
            )
            station_points.append(station_point)
        if self.reduce_points != 0:
            station_points = station_points[:-self.reduce_points]
        return station_points

    def set_reduce_points(self, num):
        result = Station(
            self.coordinate_group,
            self.color_group,
            reduce_points=num,
            **self.kwargs
        )
        return result


class AnimateStation(PersonalizedCoordinateSystem):
    CONFIG = {
        "horizontal": True,
        "show_name": True,
        "station_name_data": None,
        "show_run_time": DEFAULT_SHOW_RUN_TIME,
        "name_scale_factor": DEFAULT_NAME_SCALE_FACTOR,
        "new_station_animation": DEFAULT_NEW_STATION_ANIMATION,
        "point_animation": DEFAULT_POINT_ANIMATION,
        "grow_station_animation": DEFAULT_GROW_STATION_ANIMATION,
        "sub_point_animation": DEFAULT_SUB_POINT_ANIMATION,
        "new_station_name_animation": DEFAULT_NEW_STATION_NAME_ANIMATION,
        "interchange_station_name_animation": DEFAULT_INTERCHANGE_STATION_NAME_ANIMATION,
    }

    def __init__(self, station_coordinate, route_obj, station_name, name_direction, buff_to_center_coordinate, **kwargs):
        PersonalizedCoordinateSystem.__init__(self, **kwargs)
        global global_station_data, global_reversed_data
        assert is_np_data(station_coordinate, required_dtype="int64", required_shape=(2,))
        assert isinstance(route_obj, Route)
        self.station_coordinate = station_coordinate
        self.route_obj = route_obj
        self.station_name = station_name
        self.name_direction = name_direction
        self.buff_to_center_coordinate = buff_to_center_coordinate
        self.get_init_stuff()
        self.config_station()
        #if self.show_name():
        #    self.config_name()
        

    def get_init_stuff(self):
        self.new = True if self.horizontal is not None else False
        proportion = self.route_obj.point_on_route_proportion(self.station_coordinate)
        if proportion is None:
            raise AssertionError(self.station_coordinate)
        self.total_time = self.route_obj.total_time
        self.time_point = proportion * self.total_time
        self.start_time_points = np.array([])
        self.end_time_points = np.array([])
        self.component_animations = []
        self.station_color = self.route_obj.route_color


    def config_station(self):
        if self.new:
            self.config_new_station()
        else:
            self.config_interchange_station()

    def config_new_station(self):
        new_station = Station(
            np.array([self.station_coordinate], ndmin=2),
            [self.station_color],
            horizontal=self.horizontal
        )
        self.station_object = new_station
        
        new_name = self.get_station_name(new_station.center_coordinate)
        self.name_object = new_name
        new_station.__setattr__("station_name_obj", new_name)
        self.add_global_data(new_station)
        self.component_animations.extend([
            self.get_animation(
                self.new_station_animation,
                new_station,
                self.time_point - self.show_run_time,
                self.time_point
            ),
            self.get_animation(
                self.new_station_name_animation,
                new_name,
                self.time_point - self.show_run_time,
                self.time_point
            )
        ])
        return self

    def config_interchange_station(self):
        try:
            old_station = global_station_data[tuple(self.station_coordinate)]
        except KeyError:
            raise AssertionError(self.station_coordinate)
        self.pop_global_data(old_station)
        old_station.color_group.append(self.station_color)
        new_station = Station(
            np.r_[old_station.coordinate_group, [self.station_coordinate]],
            old_station.color_group,
            horizontal=old_station.horizontal
        )
        self.station_object = new_station
        
        new_name = self.get_station_name(new_station.center_coordinate)
        self.name_object = new_name
        old_name = old_station.station_name_obj
        new_station.__setattr__("station_name_obj", new_name)
        self.add_global_data(new_station)
        last_point = new_station.station_points[-1]
        if new_station.station_size == 2:
            transition_station = new_station.set_reduce_points(2)
            sub_point = new_station.station_points[-2]
            self.component_animations.append(self.get_animation(
                self.sub_point_animation,
                sub_point,
                self.time_point - 2 * self.show_run_time,
                self.time_point - self.show_run_time
            ))
        else:
            transition_station = new_station.set_reduce_points(1)
        self.component_animations.extend([
            self.get_animation(
                self.grow_station_animation,
                [old_station, transition_station],
                self.time_point - 2 * self.show_run_time,
                self.time_point - self.show_run_time
            ),
            self.get_animation(
                self.point_animation,
                last_point,
                self.time_point - self.show_run_time,
                self.time_point
            ),
            self.get_animation(
                self.interchange_station_name_animation,
                [old_name, new_name],
                self.time_point - 2 * self.show_run_time,
                self.time_point - self.show_run_time
            )
        ])
        return self

    def get_station_name(self, center_coordinate):
        # See mobject.py Line716 get_critical_point
        # Choose a direction from 8 of the name TextMobject,
        # then match this point to the center coordinate with a buff.
        
        #station_name, name_direction, buff_to_center_coordinate = self.name_data
        name = TextMobject(self.station_name)
        name.scale(self.name_scale_factor)
        name.move_to(
            self.change_coordinate(center_coordinate + self.buff_to_center_coordinate),
            self.name_direction
        )
        return name
        # shift & move_to

    def add_global_data(self, new_station):
        positive_direction = PLAIN_RIGHT if new_station.horizontal else PLAIN_UP
        coordinates = []
        for sign in [-1, 1]:
            coordinate = new_station.center_coordinate + (sign * (new_station.station_size + 1) * positive_direction / 2)
            coordinate = round_int(coordinate)
            global_station_data[tuple(coordinate)] = new_station
            coordinates.append(coordinate)
        global_reversed_data[(new_station)] = coordinates

    def pop_global_data(self, old_station):
        old_referred_coordinates = global_reversed_data[old_station]
        for coordinate in old_referred_coordinates:
            global_station_data.pop(tuple(coordinate))
        global_reversed_data.pop(old_station)

    def get_animation(self, animation_name, obj, a, b):
        self.start_time_points = np.append(self.start_time_points, a)
        self.end_time_points = np.append(self.end_time_points, b)
        return [animation_name, obj, a, b]

    def get_start_time_points(self):
        return self.start_time_points

    def get_end_time_points(self):
        return self.end_time_points

    def get_animations(self):
        return self.component_animations

    def get_mobject(self):
        return [self.station_object, self.name_object]


class TimeScene(Scene):
    # About kwargs...
    def get_stuff_and_play(self, route_data, stations_data, route_color):
        route = Route(route_data, route_color=route_color)
        stations = [
            AnimateStation(coordinate, route, name, name_direction, buff, horizontal=horizontal)
            for coordinate, horizontal, name, name_direction, buff in stations_data
        ]
        self.play_route(route, stations)
        self.add(route)
        for station in stations:
            self.add(*station.get_mobject())

    def play_route(self, route, animate_stations, **kwargs):
        assert isinstance(route, Route) and not isinstance(route, StationFrame)
        assert all([isinstance(animate_station, AnimateStation) for animate_station in animate_stations])
        self.total_time = route.total_time
        self.animate_stations = animate_stations
        lagged_time, delayed_time = self.adjust_time()
        animations = self.get_station_animations()
        self.total_time += (lagged_time + delayed_time)
        self.play(
            ShowCreation(
                route,
                run_time=self.total_time,
                rate_func=route.get_route_linear_func(
                    a=lagged_time / self.total_time,
                    b=1 - delayed_time / self.total_time
                ),
                **kwargs
            ),
            *[
                animation_name(obj, self.total_time, a + lagged_time, b + lagged_time)
                for animation_name, obj, a, b in animations
            ]
        )

    def adjust_time(self):
        start_time_points_list = np.array([])
        end_time_points_list = np.array([])
        for animate_station in self.animate_stations:
            start_time_points_list = np.append(start_time_points_list, animate_station.get_start_time_points())
            end_time_points_list = np.append(end_time_points_list, animate_station.get_end_time_points())
        lagged_time = -min(np.min(start_time_points_list), 0)
        delayed_time = max(np.max(end_time_points_list) - self.total_time, 0)
        return lagged_time, delayed_time

    def get_station_animations(self):
        animations = []
        for animate_station in self.animate_stations:
            animations.extend(animate_station.component_animations)
        return animations


example_route_coordinates1 = np_int(
    [30, 10],
    [30, 35],
    [45, 50],
    [55, 50],
    [70, 35],
    [75, 35],
    [75, 55],
    [80, 55],
    [90, 45],
    [130, 45],
)
example_stations1 = [
    [np_int(30, 10), True, "Station 01", LEFT, np.array([1.0, 0.0])],
    [np_int(30, 25), False, "Station 02", LEFT, np.array([1.0, 0.0])],
    [np_int(37, 42), True, "Station 03", UL, np.array([0.8, -0.8])],
    [np_int(50, 50), True, "Station 04", DOWN, np.array([0.0, 1.0])],
    [np_int(64, 41), True, "Station 05", DL, np.array([0.8, 0.8])],
    [np_int(72, 35), False, "Station 06", UP, np.array([0.0, -1.0])],
    [np_int(78, 55), True, "Station 07", DOWN, np.array([0.0, 1.0])],
    [np_int(84, 51), True, "Station 08", DL, np.array([0.8, 0.8])],
    [np_int(92, 45), True, "Station 09", UP, np.array([0.0, -1.0])],
    [np_int(121, 45), True, "Station 10", UP, np.array([0.0, -1.0])],
    [np_int(130, 45), True, "Station 11", UP, np.array([0.0, -1.0])],
]
example_route_coordinates2 = np_int(
    [30, 55],
    [30, 48],
    [44, 34],
    [93, 34],
    [93, 55],
)
example_stations2 = [
    [np_int(30, 55), True, "Station 12", LEFT, np.array([1.0, 0.0])],
    [np_int(36, 42), None, "Station 03", LEFT, np.array([1.5, 0.0])],
    [np_int(56, 34), False, "Station 13", UP, np.array([0.0, -1.0])],
    [np_int(72, 34), None, "Station 06", UP, np.array([0.0, -1.5])],
    [np_int(84, 34), True, "Station 14", UP, np.array([0.0, -1.0])],
    [np_int(93, 45), None, "Station 09", UR, np.array([0.0, -1.0])],
    [np_int(93, 55), False, "Station 15", LEFT, np.array([1.0, 0.0])],
]
example_route_coordinates3 = np_int(
    [108, 54],
    [86, 54],
    [72, 40],
    [72, 26],
    [23, 26],
)
example_stations3 = [
    [np_int(108, 54), True, "Station 16", UP, np.array([0.0, -1.0])],
    [np_int(93, 54), None, "Station 15", DL, np.array([1.0, 0.0])],
    [np_int(83, 51), None, "Station 08", UP, np.array([0.0, -1.0])],
    [np_int(72, 36), None, "Station 06", UP, np.array([0.0, -2.0])],
    [np_int(50, 26), True, "Station 17", UP, np.array([0.0, -1.0])],
    [np_int(30, 26), None, "Station 02", UL, np.array([1.0, 0.0])],
    [np_int(23, 26), True, "Station 18", UP, np.array([0.0, -1.0])],
]
example_route_coordinates4 = np_int(
    [38, 42],
    [38, 35],
    [61, 35],
    [63, 33],
    [82, 33],
    [90, 25],
)
example_stations4 = [
    [np_int(38, 42), None, "Station 03", LEFT, np.array([2.0, 0.0])],
    [np_int(56, 35), None, "Station 13", UP, np.array([0.0, -1.5])],
    [np_int(72, 33), None, "Station 06", UP, np.array([0.0, -2.5])],
    [np_int(90, 25), True, "Station 19", UP, np.array([0.0, -1.0])],
]
example_route_coordinates5 = np_int(
    [72, 32],
    [91, 32],
    [91, 51],
    [70, 51],
)
example_stations5 = [
    [np_int(72, 32), None, "Station 06", UP, np.array([0.0, -3.0])],
    [np_int(91, 45), None, "Station 09", UP, np.array([0.0, -1.0])],
    [np_int(85, 51), None, "Station 08", UP, np.array([0.0, -1.0])],
    [np_int(70, 51), True, "Station 20", DOWN, np.array([0.0, 1.0])],
]


class ExpScene(TimeScene):
    def construct(self):
        self.get_stuff_and_play(example_route_coordinates1, example_stations1, BLUE_B)
        self.wait()
        self.get_stuff_and_play(example_route_coordinates2, example_stations2, RED)
        self.wait()
        self.get_stuff_and_play(example_route_coordinates3, example_stations3, GREEN)
        self.wait()
        self.get_stuff_and_play(example_route_coordinates4, example_stations4, YELLOW)
        self.wait()
        self.get_stuff_and_play(example_route_coordinates5, example_stations5, PURPLE)
        self.wait(3)

