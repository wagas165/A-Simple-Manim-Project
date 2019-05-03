from big_ol_pile_of_manim_imports import *
from copy import deepcopy
from pprint import pprint

# TODO, record the lagged time and make the time shaft.
# Find or make the settings of layers.
# See leibniz.py line 583. Make use of VGroup.
# Better to have a class of all Config.

# TODO, find a way to let the station be above the lines.
# Try to combine the classes to decrease the reset of attributes.
# PACK THE CLASSES!!!!


DEFAULT_ROUTE_WIDTH = 8
DEFAULT_ROUTE_ORIGINAL_RADIUS_RATIO = 3 # the ratio of the inner radius and half of the width
DEFAULT_PLAY_SPEED = 15.0 # widths per second
DEFAULT_ROUTE_OPACITY = 0.6######
#DEFAULT_OUTER_RING_WIDTH_FACTOR = 1.25
#DEFAULT_INNER_RING_WIDTH_FACTOR = 0.75
DEFAULT_FADE_FACTOR = 3.0
DEFAULT_NEW_STATION_STROKE_WIDTH = 3
DEFAULT_INTERCHANGE_STATION_STROKE_WIDTH = 1
DEFAULT_POINT_RADIUS_RATIO = 0.8
DEFAULT_NEW_STATION_RADIUS_RATIO = 1.0
DEFAULT_INTERCHANGE_STATION_RADIUS_RATIO = 1.2
DEFAULT_SHOW_RUN_TIME = 1.0


def np_int(*elements):
    return np.array(elements, dtype="int64")

PLAIN_RIGHT = np_int(1, 0)
PLAIN_LEFT = np_int(-1, 0)
PLAIN_UP = np_int(0, 1)
PLAIN_DOWN = np_int(0, -1)


def get_direction(horizontal, positive_direction):
    if horizontal:
        direction = PLAIN_RIGHT if positive_direction else PLAIN_LEFT
    else:
        direction = PLAIN_UP if positive_direction else PLAIN_DOWN
    return direction


# The parameter required_shape should be a tuple which consists positive integers or None.
# e.g. required_shape=(3, 1, None)
def is_np_data(data, required_dtype=None, required_shape=None):
    if type(data) != np.ndarray:
        return False
    if required_dtype == None:
        if "int" not in str(data.dtype) and "float" not in str(data.dtype):
            return False
    elif required_dtype != str(data.dtype):
        return False
    if required_shape == None:
        return True
    data_shape = np.shape(data)
    required_dimension = len(required_shape)
    if len(data_shape) != required_dimension:
        return False
    for k in range(required_dimension):
        if required_shape[k] != None and data_shape[k] != required_shape[k]:
            return False
    return True


def change_angle(angle):
    return angle * PI / 8


# The rate between unit and stroke_width is simply 100:1.
def change_unit(length, unit_length=1):
    return 0.01 * length * unit_length


# The origin is in the bottom left corner of the screen.
def change_coordinate(two_d_coordinate, unit_length):
    unit_stroke_width = change_unit(unit_length)
    x = unit_stroke_width * two_d_coordinate[0] - FRAME_X_RADIUS
    y = unit_stroke_width * two_d_coordinate[1] - FRAME_Y_RADIUS
    return np.array([x, y, 0])


# Here, the route should be standardized.
class ConfigRoute(object):
    CONFIG = {
       "name": None,
        "color": WHITE,
        "route_width": DEFAULT_ROUTE_WIDTH,
        "route_original_radius_ratio": DEFAULT_ROUTE_ORIGINAL_RADIUS_RATIO,
        "route_opacity": DEFAULT_ROUTE_OPACITY,
        "special_original_radius_ratios": np.array([], dtype="int64"),
        "play_speed": DEFAULT_PLAY_SPEED,
        "animate_func": ShowCreation,
        "rate_func": linear,
    }

    def __init__(self, orderd_coordinates, **kwargs):
        digest_config(self, kwargs)
        self.orderd_coordinates = orderd_coordinates
        self.get_ordered_coordinates()
        self.get_original_radius_ratios()
        self.get_radius_ratios()
        self.get_distances()
        self.get_directions()
        self.get_turning_angles()
        self.get_reduced_line_lengths()
        self.get_route_components()
        self.get_time_points()

    def get_ordered_coordinates(self):
        assert is_np_data(self.orderd_coordinates, required_dtype="int64", required_shape=(None, 2))
        self.num_of_lines = np.shape(self.orderd_coordinates)[0] - 1
        assert self.num_of_lines > 0

    def get_original_radius_ratios(self):
        self.original_radius_ratios = np.ones(self.num_of_lines, dtype="int64") * self.route_original_radius_ratio
        assert np.size(self.special_original_radius_ratios) == 0 or is_np_data(self.special_original_radius_ratios, required_dtype="int64", required_shape=(None, 2))
        for k in self.special_original_radius_ratios:
            assert k[1] >= 0
            self.original_radius_ratios[k[0]] = k[1]

    def get_radius_ratios(self):
        self.radius_ratios = (self.original_radius_ratios + 1) / 2

    def get_distances(self):
        self.distances = np.zeros(self.num_of_lines, dtype="float64")
        for k in range(self.num_of_lines):
            distance = get_norm(self.orderd_coordinates[k] - self.orderd_coordinates[k+1])
            assert distance > 0
            self.distances[k] = distance

    def get_directions(self):
        self.directions = np.zeros(self.num_of_lines, dtype="int8")
        for k in range(self.num_of_lines):
            x0, y0 = self.orderd_coordinates[k]
            x1, y1 = self.orderd_coordinates[k+1]
            if x0 == x1:
                direction = 4 if y0 < y1 else 12
            elif y0 == y1:
                direction = 0 if x0 < x1 else 8
            elif x1 - x0 == y1 - y0:
                direction = 2 if x0 < x1 else 10
            elif x1 - x0 == y0 - y1:
                direction = 6 if x0 > x1 else 14
            else:
                raise AssertionError
            self.directions[k] = direction
            if k != 0:
                turning_direction = abs(self.directions[k] - self.directions[k-1])
                assert turning_direction not in {6, 10}

    def get_turning_angles(self):
        self.turning_angles = np.zeros(self.num_of_lines, dtype="int8")
        for k in range(1, self.num_of_lines):
            turning_angle = (self.directions[k-1] - self.directions[k] + 8) % 16
            if turning_angle > 8:
                turning_angle -= 16
            self.turning_angles[k] = turning_angle

    def get_reduced_line_lengths(self):
        self.reduced_line_lengths = np.zeros(self.num_of_lines+1, dtype="float64")
        for k in range(1, self.num_of_lines):
            assert self.turning_angles[k] not in {0, 8}
            reduced_line_length = self.radius_ratios[k] / np.tan(change_angle(abs(self.turning_angles[k]) / 2))
            self.reduced_line_lengths[k] = reduced_line_length

    def get_route_components(self):
        self.components = []
        self.start_coordinates = np.zeros((self.num_of_lines, 2), dtype="float64")
        self.end_coordinates = np.zeros((self.num_of_lines, 2), dtype="float64")
        for k in range(self.num_of_lines):
            self.get_bended_route(k)
            self.get_straight_route(k)
        self.num_of_components = len(self.components)

    def get_straight_route(self, k):
        straight_route_lenth = self.distances[k] - (self.reduced_line_lengths[k] + self.reduced_line_lengths[k+1])
        assert straight_route_lenth >= 0
        start = np.array([
            self.orderd_coordinates[k][0] + self.reduced_line_lengths[k] * np.cos(change_angle(self.directions[k])),
            self.orderd_coordinates[k][1] + self.reduced_line_lengths[k] * np.sin(change_angle(self.directions[k]))
        ])
        end = np.array([
            self.orderd_coordinates[k+1][0] - self.reduced_line_lengths[k+1] * np.cos(change_angle(self.directions[k])),
            self.orderd_coordinates[k+1][1] - self.reduced_line_lengths[k+1] * np.sin(change_angle(self.directions[k]))
        ])
        self.start_coordinates[k] = start
        self.end_coordinates[k] = end
        straight_route = StraightRoute(start, end, self.route_width, self.route_opacity, self.color)
        self.components.append(straight_route)

    def get_bended_route(self, k):
        if k != 0:
            bisector_argument = self.directions[k-1] - self.turning_angles[k] // 2
            while bisector_argument > 8:
                bisector_argument -= 16
            center_distance = self.radius_ratios[k] / np.sin(change_angle(abs(self.turning_angles[k]) / 2))
            arc_center = np.array([
                self.orderd_coordinates[k][0] - np.cos(change_angle(bisector_argument)) * center_distance,
                self.orderd_coordinates[k][1] - np.sin(change_angle(bisector_argument)) * center_distance
            ])
            radius = self.radius_ratios[k]
            if self.turning_angles[k] > 0:
                start_angle = change_angle(self.directions[k-1] - 4)
                angle = change_angle(-self.turning_angles[k] + 8)
            elif self.turning_angles[k] < 0:
                start_angle = change_angle(self.directions[k-1] + 4)
                angle = change_angle(-self.turning_angles[k] - 8)
            bended_route = BendedRoute(arc_center, start_angle, angle, radius, self.route_width, self.route_opacity, self.color)
            self.components.append(bended_route)

    def get_time_points(self):
        self.time_points = np.zeros(self.num_of_components+1, dtype="float64")
        self.before_straight_time_points = np.array([0], dtype="float64")
        self.after_straight_time_points = np.array([], dtype="float64")
        for k in range(self.num_of_components):
            component = self.components[k]
            time_point = self.time_points[k] + component.get_init_length()
            self.time_points[k+1] = time_point
            if isinstance(component, BendedRoute):
                previous_time_point = self.time_points[k]
                self.before_straight_time_points = np.append(self.before_straight_time_points, [time_point])
                self.after_straight_time_points = np.append(self.after_straight_time_points, [previous_time_point])
        self.total_time = self.time_points[-1] / self.play_speed
        self.start_time_points = self.time_points[:-1]
        self.end_time_points = self.time_points[1:]
        self.time_points /= self.play_speed
        self.before_straight_time_points /= self.play_speed
        self.after_straight_time_points /= self.play_speed
        self.after_straight_time_points = np.append(self.after_straight_time_points, [self.total_time])
        assert self.num_of_lines == len(self.before_straight_time_points) == len(self.after_straight_time_points)
        self.animate_function = [self.animate_func] * self.num_of_components
        self.rate_function = [self.rate_func] * self.num_of_components


class StraightRoute(Line):
    def __init__(self, start, end, width, opacity, color, **kwargs):
        self.init_start = start
        self.init_end = end
        self.stroke_width = width
        self.stroke_opacity = opacity
        self.color = color
        super().__init__(change_coordinate(start, width), change_coordinate(end, width), **kwargs)

    def get_init_length(self):
        return get_norm(self.init_start - self.init_end)


class BendedRoute(Arc):
    def __init__(self, arc_center, start_angle, angle, radius, width, opacity, color, **kwargs):
        self.init_radius = radius
        self.stroke_width = width
        self.arc_center = change_coordinate(arc_center, width)
        self.radius = 0.01 * radius * width
        self.stroke_opacity = opacity
        self.color = color
        super().__init__(start_angle, angle, **kwargs)

    def get_init_length(self):
        return abs(self.angle) * self.init_radius


######

# Make the data globally. For each of the station, there are two keys concerned.
# dict_key: the to-extend coordinate of the station and positive_direction as a boolean as a tuple
# dict_value: a complete station as a VGroup
global_station_backgrounds_data = {}

# Config a single station.
class ConfigStation(object):
    CONFIG = {
        "name": None,
        "station_color": WHITE,
        "background_color": BLACK,
        "station_stroke_width": None,##
        "station_stroke_color": None,#
        "station_radius_ratio": None,##
        "center_coordinate": None,
        "show_run_time": DEFAULT_SHOW_RUN_TIME,
    }

    def __init__(self, station_coordinate, route_obj, **kwargs):
        digest_config(self, kwargs)
        global global_station_backgrounds_data
        self.station_coordinate = station_coordinate
        self.route_obj = route_obj
        self.get_station_coordinate()
        self.get_station_time_point()

    def get_station_coordinate(self):
        assert isinstance(self.route_obj, ConfigRoute)
        assert is_np_data(self.station_coordinate, required_dtype="int64", required_shape=(2,))
        self.route_width = self.route_obj.route_width

    def get_station_time_point(self):
        route = self.route_obj
        self.total_time = route.total_time
        self.route_width = route.route_width
        self.station_color = route.color
        x, y = self.station_coordinate
        for k in range(route.num_of_lines):
            x0, y0 = route.orderd_coordinates[k]
            x1, y1 = route.orderd_coordinates[k+1]
            if (x * (y0 - y1) + x0 * y1 == y * (x0 - x1) + x1 * y0) and ((x - x0) * (x - x1) <= 0):
                z, z0, z1 = complex(x, y), complex(x0, y0), complex(x1, y1)
                if abs(z - z0) >= route.reduced_line_lengths[k] and abs(z - z1) >= route.reduced_line_lengths[k+1]:
                    w0, w1 = complex(route.start_coordinates[k][0], route.start_coordinates[k][1]), complex(route.end_coordinates[k][0], route.end_coordinates[k][1])
                    t0, t1 = route.before_straight_time_points[k], route.after_straight_time_points[k]
                    time_point = t0 + abs(z - w0) / abs(w1 - w0) * (t1 - t0)
                    break
        else:
            print(self.station_coordinate)
            raise AssertionError
        self.key_time_point = time_point


class ConfigNewStation(ConfigStation):
    CONFIG = {
        "horizontal": True,
        "station_stroke_width": DEFAULT_NEW_STATION_STROKE_WIDTH,
        "station_stroke_color": None,
        "station_radius_ratio": DEFAULT_NEW_STATION_RADIUS_RATIO,
        #"outer_ring_width_factor": DEFAULT_OUTER_RING_WIDTH_FACTOR,
        #"inner_ring_width_factor": DEFAULT_INNER_RING_WIDTH_FACTOR,
        "animate_func": lambda mobject, **kwargs: FadeInFromLarge(mobject, scale_factor=DEFAULT_FADE_FACTOR, **kwargs),
        "rate_func": smooth,
    }

    def __init__(self, station_coordinate, route_obj, **kwargs):
        super().__init__(station_coordinate, route_obj, **kwargs)
        self.center_coordinate = station_coordinate
        self.station_size = 1
        self.create_station()

    def create_station(self):
        station = StationGroup(SetStationBackground(self.station_coordinate, 1, self.station_radius_ratio, self.route_width, self.station_stroke_width, self.station_color, self.background_color, self.horizontal))
        self.components = [station.station_components]
        if self.horizontal:
            neighbour_coordinates = [tuple(self.station_coordinate + PLAIN_RIGHT), tuple(self.station_coordinate + PLAIN_LEFT)]
        else:
            neighbour_coordinates = [tuple(self.station_coordinate + PLAIN_UP), tuple(self.station_coordinate + PLAIN_DOWN)]
        global_station_backgrounds_data[(*neighbour_coordinates[0], True)] = station
        global_station_backgrounds_data[(*neighbour_coordinates[1], False)] = station
        self.animate_function = [self.animate_func]
        self.rate_function = [self.rate_func]
        self.end_time_points = [self.key_time_point]
        self.start_time_points = [self.key_time_point - self.show_run_time]
        self.num_of_components = 1


class ConfigInterchangeStation(ConfigStation):
    CONFIG = {
        "positive_direction": True,
        "station_stroke_width": DEFAULT_INTERCHANGE_STATION_STROKE_WIDTH,
        "station_stroke_color": WHITE,
        "station_radius_ratio": DEFAULT_INTERCHANGE_STATION_RADIUS_RATIO,
        "point_radius_ratio": DEFAULT_POINT_RADIUS_RATIO,
        #"outer_ring_width_factor": DEFAULT_OUTER_RING_WIDTH_FACTOR,
        #"inner_ring_width_factor": DEFAULT_INNER_RING_WIDTH_FACTOR,
        "animate_func_background": lambda tuple_of_mobjects, **kwargs: Transform(*tuple_of_mobjects, replace_mobject_with_target_in_scene=True, **kwargs),
        "rate_func_background": double_smooth,
        "animate_func_append_point": lambda mobject, **kwargs: FadeInFromLarge(mobject, scale_factor=DEFAULT_FADE_FACTOR, **kwargs),
        "rate_func_append_point": smooth,
        "animate_func_old_point": GrowFromCenter,
        "rate_func_old_point": linear,
    }

    def __init__(self, station_coordinate, route_obj, **kwargs):
        super().__init__(station_coordinate, route_obj, **kwargs)
        #self.center_coordinate
        self.get_old_station()
        self.grow_station()

    def get_old_station(self):
        key = (*self.station_coordinate, self.positive_direction)
        try:
            self.old_station = global_station_backgrounds_data[key]
        except KeyError:
            raise AssertionError(key)
        self.old_station = global_station_backgrounds_data[key]
        global_station_backgrounds_data.pop(key)
        direction = get_direction(self.old_station.horizontal, self.positive_direction)
        another_coordinate = self.station_coordinate - (self.old_station.station_size + 1) * direction
        another_key = (*another_coordinate, not self.positive_direction)
        global_station_backgrounds_data.pop(another_key)
        new_coordinate = self.station_coordinate + direction
        new_key = (*new_coordinate, self.positive_direction)
        self.new_dict_keys = [another_key, new_key]
        self.center_coordinate = self.old_station.center_coordinate + direction / 2
        self.horizontal = self.old_station.horizontal
        self.station_size = self.old_station.station_size + 1

    def grow_station(self):
        run_time = self.show_run_time
        self.new_station = self.grow_station_background(self.old_station, self.positive_direction)
        self.new_point = SetPoint(self.station_coordinate, self.station_color, self.route_width, self.point_radius_ratio)
        self.components = [(self.old_station, self.new_station), self.new_point]
        #self.components = [(copy.deepcopy(self.old_station), copy.deepcopy(self.new_station)), self.new_point]#
        self.animate_function = [self.animate_func_background, self.animate_func_append_point]
        self.rate_function = [self.rate_func_background, self.rate_func_append_point]
        self.start_time_points = [self.key_time_point - 2 * run_time, self.key_time_point - run_time]
        self.end_time_points = [self.key_time_point - run_time, self.key_time_point]
        if self.station_size == 2:
            self.init_color = self.old_station.init_color
            self.old_point = SetPoint(self.old_station.center_coordinate, self.init_color, self.route_width, self.point_radius_ratio)
            self.components.append(self.old_point)
            self.animate_function.append(self.animate_func_old_point)
            self.rate_function.append(self.rate_func_old_point)
            self.start_time_points.append(self.key_time_point - run_time)
            self.end_time_points.append(self.key_time_point)
            self.num_of_components = 3
            #composed_station = StationGroup(self.new_station.background_part, self.old_point, self.new_point)
            self.new_station.points_part = [self.old_point, self.new_point]

            #self.new_station = StationGroup(self.new_station.background_part, *self.new_station.points_part)#
            composed_station = StationGroup(self.new_station.background_part, *self.new_station.points_part)
            #composed_station = self.new_station.add(self.old_point, self.new_point)
        else:
            self.num_of_components = 2
            self.new_station.points_part.append(self.new_point)
            #self.new_station = StationGroup(self.new_station.background_part, *self.new_station.points_part)#
            composed_station = StationGroup(self.new_station.background_part, *self.new_station.points_part)
            #print(composed_station.points_part)
            #self.new_station.points_part.append(self.new_point)
            #points = []
            #for point in self.new_station.points_part:
            #    points.append(point)
            #points.append(self.new_point)
            #print(points)
            #print(self.new_station.points_part)
            #composed_station = StationGroup(self.new_station.background_part, *points)
            #composed_station = self.new_station.add(self.new_point)
        for key in self.new_dict_keys:
            global_station_backgrounds_data[key] = composed_station

    def grow_station_background(self, old_station_obj, positive_direction):
        assert isinstance(old_station_obj, StationGroup)
        horizontal = old_station_obj.horizontal
        direction = get_direction(horizontal, positive_direction)
        center_coordinate = old_station_obj.center_coordinate + direction / 2
        station_size = old_station_obj.station_size + 1
        #station_radius_ratio = old_station_obj.station_radius_ratio
        #route_width = old_station_obj.route_width
        #station_stroke_width = old_station_obj.station_stroke_width
        #station_stroke_color = old_station_obj.station_stroke_color
        #background_color = old_station_obj.background_color
        self.background_part = SetStationBackground(center_coordinate, station_size, self.station_radius_ratio, self.route_width, self.station_stroke_width, self.station_stroke_color, self.background_color, horizontal)
        self.points_part = old_station_obj.points_part
        return StationGroup(self.background_part, *self.points_part)


class SetPoint(Circle):
    def __init__(self, center_coordinate, color, route_width, radius_ratio, **kwargs):
        super().__init__(arc_center=change_coordinate(center_coordinate, route_width), radius=change_unit(route_width * radius_ratio / 2), fill_color=color, fill_opacity=1, stroke_width=0, **kwargs)


class SetStationBackground(VGroup):
    def __init__(self, center_coordinate, station_size, station_radius_ratio, route_width, station_stroke_width, station_stroke_color, background_color, horizontal, **kwargs):
        ##
        self.center_coordinate = center_coordinate
        self.station_size = station_size
        #self.station_radius_ratio = station_radius_ratio
        #self.route_width = route_width
        #self.station_stroke_width = station_stroke_width
        self.station_stroke_color = station_stroke_color
        #self.background_color = background_color
        self.horizontal = horizontal
        ##
        radius = station_radius_ratio / 2
        half_width = (station_size - 1) / 2
        change = lambda coordinate: change_coordinate(coordinate, route_width)
        unify = lambda length: change_unit(length, route_width)
        if horizontal:
            right_center = center_coordinate + half_width * PLAIN_RIGHT
            left_center = center_coordinate + half_width * PLAIN_LEFT
            self.background_components = [
                Rectangle(height=unify(2 * radius), width=unify(2 * half_width), stroke_width=0, fill_color=background_color, fill_opacity=0.5).move_to(change(center_coordinate)),
                Line(change(right_center + radius * PLAIN_UP), change(left_center + radius * PLAIN_UP), stroke_width=station_stroke_width, stroke_color=station_stroke_color, stroke_opacity=1),
                Arc(start_angle=PI / 2, angle=PI, arc_center=change(left_center), radius=unify(radius), fill_color=background_color, fill_opacity=0.5, stroke_width=station_stroke_width, stroke_color=station_stroke_color, stroke_opacity=1),
                Line(change(left_center + radius * PLAIN_DOWN), change(right_center + radius * PLAIN_DOWN), stroke_width=station_stroke_width, stroke_color=station_stroke_color, stroke_opacity=1),
                Arc(start_angle=PI * 3 / 2, angle=PI, arc_center=change(right_center), radius=unify(radius), fill_color=background_color, fill_opacity=0.5, stroke_width=station_stroke_width, stroke_color=station_stroke_color, stroke_opacity=1),
            ]
        else:
            up_center = center_coordinate + half_width * PLAIN_UP
            down_center = center_coordinate + half_width * PLAIN_DOWN
            self.background_components = [
                Rectangle(height=unify(2 * half_width), width=unify(2 * radius), stroke_width=0, fill_color=background_color, fill_opacity=0.5).move_to(change(center_coordinate)),
                Arc(start_angle=0, angle=PI, arc_center=change(up_center), radius=unify(radius), fill_color=background_color, fill_opacity=0.5, stroke_width=station_stroke_width, stroke_color=station_stroke_color, stroke_opacity=1),
                Line(change(up_center + radius * PLAIN_LEFT), change(down_center + radius * PLAIN_LEFT), stroke_width=station_stroke_width, stroke_color=station_stroke_color, stroke_opacity=1),
                Arc(start_angle=PI, angle=PI, arc_center=change(down_center), radius=unify(radius), fill_color=background_color, fill_opacity=0.5, stroke_width=station_stroke_width, stroke_color=station_stroke_color, stroke_opacity=1),
                Line(change(down_center + radius * PLAIN_RIGHT), change(up_center + radius * PLAIN_RIGHT), stroke_width=station_stroke_width, stroke_color=station_stroke_color, stroke_opacity=1),
            ]
        super().__init__(*self.background_components, **kwargs)


class StationGroup(VGroup):
    def __init__(self, background_obj, *point_objs):
        assert isinstance(background_obj, SetStationBackground)
        for obj in point_objs:
            assert isinstance(obj, SetPoint)
        self.center_coordinate = background_obj.center_coordinate
        self.station_size = background_obj.station_size
        #self.station_radius_ratio = background_obj.station_radius_ratio
        #self.route_width = background_obj.route_width
        #self.station_stroke_width = background_obj.station_stroke_width
        self.station_stroke_color = background_obj.station_stroke_color
        #self.background_color = background_obj.background_color
        self.horizontal = background_obj.horizontal
        if self.station_size == 1:
            self.init_color = background_obj.station_stroke_color
        self.background_part = background_obj
        self.points_part = list(point_objs)
        #for point in point_objs:
        #    background_obj.station_components.append(point)
        complete_station_components = deepcopy(self.background_part.background_components)
        #print("-----")
        #print(len(point_objs))
        #pprint(complete_station_components)
        #self.background_part.background_components.extend(self.points_part)
        #pprint(self.background_part.background_components)
        complete_station_components.extend(self.points_part)
        #pprint(self.points_part)
        #pprint(complete_station_components)
        self.station_components = VGroup(*complete_station_components)
        super().__init__(*self.station_components)


########
# The *config_mobjects should have these attributes:
# components(list), animate_function(func), rate_function(func), total_time(num), start_time_points(list), end_time_points(list), num_of_components(int).
class TimeScene(Scene):
    def play_route(self, *config_mobjects):
        arguments = self.get_all_arguments_zipped(config_mobjects)
        self.play(*[
            animate(component, run_time=total, rate_func=squish_rate_func(rate, start, end))
            for component, animate, rate, total, start, end in arguments
        ])

    def get_all_arguments_zipped(self, config_mobjects):
        all_components = []
        for mobject in config_mobjects:
            all_components.extend(mobject.components)
        num_of_all_components = len(all_components)
        all_animations = []
        all_rates = []
        all_total_time = np.zeros(num_of_all_components, dtype="float64")
        all_start_time_points = np.zeros(num_of_all_components, dtype="float64")
        all_end_time_points = np.zeros(num_of_all_components, dtype="float64")
        index = 0
        for mobject in config_mobjects:
            total = mobject.total_time
            all_animations.extend(mobject.animate_function)
            all_rates.extend(mobject.rate_function)
            for k in range(mobject.num_of_components):
                all_total_time[index] = total
                all_start_time_points[index] = mobject.start_time_points[k]
                all_end_time_points[index] = mobject.end_time_points[k]
                assert all_start_time_points[index] <= all_end_time_points[index]
                index += 1
        if min(all_start_time_points) < 0:
            delay = -min(all_start_time_points)
            all_total_time += delay
            all_start_time_points += delay
            all_end_time_points += delay
        return zip(all_components, all_animations, all_rates, all_total_time, all_start_time_points / all_total_time, all_end_time_points / all_total_time)


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
example_route1 = ConfigRoute(example_route_coordinates1, color=BLUE_B)
example_stations1 = [
    ConfigNewStation(np_int(30, 10), example_route1),
    ConfigNewStation(np_int(30, 25), example_route1, horizontal=False),
    ConfigNewStation(np_int(37, 42), example_route1),
    ConfigNewStation(np_int(50, 50), example_route1),
    ConfigNewStation(np_int(64, 41), example_route1),
    ConfigNewStation(np_int(72, 35), example_route1, horizontal=False),
    ConfigNewStation(np_int(78, 55), example_route1),
    ConfigNewStation(np_int(84, 51), example_route1),
    ConfigNewStation(np_int(92, 45), example_route1),
    ConfigNewStation(np_int(121, 45), example_route1),
    ConfigNewStation(np_int(130, 45), example_route1),
]
example_route_coordinates2 = np_int(
    [30, 55],
    [30, 48],
    [44, 34],
    [93, 34],
    [93, 55],
)
example_route2 = ConfigRoute(example_route_coordinates2, color=RED)
example_stations2 = [
    ConfigNewStation(np_int(30, 55), example_route2),
    ConfigInterchangeStation(np_int(36, 42), example_route2, positive_direction=False),
    ConfigNewStation(np_int(56, 34), example_route2, horizontal=False),
    ConfigInterchangeStation(np_int(72, 34), example_route2, positive_direction=False),
    ConfigNewStation(np_int(84, 34), example_route2),
    ConfigInterchangeStation(np_int(93, 45), example_route2, positive_direction=True),
    ConfigNewStation(np_int(93, 55), example_route2, horizontal=False),
]
example_route_coordinates3 = np_int(
    [108, 54],
    [86, 54],
    [72, 40],
    [72, 26],
    [23, 26],
)
example_route3 = ConfigRoute(example_route_coordinates3, color=GREEN)
example_stations3 = [
    ConfigNewStation(np_int(108, 54), example_route3),
    ConfigInterchangeStation(np_int(93, 54), example_route3, positive_direction=False),
    ConfigInterchangeStation(np_int(83, 51), example_route3, positive_direction=False),
    ConfigInterchangeStation(np_int(72, 36), example_route3, positive_direction=True),
    ConfigNewStation(np_int(50, 26), example_route3),
    ConfigInterchangeStation(np_int(30, 26), example_route3, positive_direction=True),
    ConfigNewStation(np_int(23, 26), example_route3),
]
example_route_coordinates4 = np_int(
    [38, 42],
    [38, 35],
    [61, 35],
    [63, 33],
    [82, 33],
    [90, 25],
)
example_route4 = ConfigRoute(example_route_coordinates4, color=YELLOW)
example_stations4 = [
    ConfigInterchangeStation(np_int(38, 42), example_route4, positive_direction=True),
    ConfigInterchangeStation(np_int(56, 35), example_route4, positive_direction=True),
    ConfigInterchangeStation(np_int(72, 33), example_route4, positive_direction=False),
    ConfigNewStation(np_int(90, 25), example_route4),
]
example_route_coordinates5 = np_int(
    [72, 32],
    [91, 32],
    [91, 51],
    [70, 51],
)
example_route5 = ConfigRoute(example_route_coordinates5, color=PURPLE)
example_stations5 = [
    ConfigInterchangeStation(np_int(72, 32), example_route5, positive_direction=False),
    ConfigInterchangeStation(np_int(91, 45), example_route5, positive_direction=False),
    ConfigInterchangeStation(np_int(85, 51), example_route5, positive_direction=True),
    ConfigNewStation(np_int(70, 51), example_route5),
]

class DrawLine(TimeScene):
    def construct(self):
        self.play_route(example_route1, *example_stations1)
        self.wait(1)
        self.play_route(example_route2, *example_stations2)
        self.wait(1)
        self.play_route(example_route3, *example_stations3)
        self.wait(1)
        self.play_route(example_route4, *example_stations4)
        self.wait(1)
        self.play_route(example_route5, *example_stations5)
        self.wait(3)
        
