"""A moving traffic window and safety-aware rewards on HighwayEnv."""

import numpy as np

from highway_env.envs.highway_env import HighwayEnv
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.behavior import IDMVehicle
from gymnasium import spaces
from highway_env.envs.common.observation import KinematicObservation
from highway_env import utils









class LaneSpeedIDMVehicle(IDMVehicle):
    """随机调整目标速度、尝试安全变道，少量车辆允许速度违规。"""

    def __init__(
        self,
        *args,
        lane_speed_ranges,
        violation_probability=0.05,
        lane_change_probability=0.25,
        event_interval=(4.0, 8.0),
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.lane_speed_ranges = lane_speed_ranges
        self.lane_change_probability = lane_change_probability
        self.event_interval = event_interval

        # 车辆生成时确定类型：守规则车辆或速度违规车辆。
        self.is_rule_breaker = bool(
            self.road.np_random.random() < violation_probability
        )

        self.previous_lane_id = self.lane_index[2]
        self.event_timer = self._next_interval()

        # 使用下面的随机变道逻辑，关闭原来的自动变道决策。
        self.enable_lane_change = False

        # 记录行为次数，方便确认随机事件确实发生了。
        self.speed_event_count = 0
        self.lane_change_request_count = 0

        self._sample_target_speed(self.previous_lane_id)

    def _next_interval(self):
        """随机决定距离下一次事件还有多少秒。"""
        return float(
            self.road.np_random.uniform(*self.event_interval)
        )

    def _sample_target_speed(self, lane_id):
        """调整目标速度，不直接修改实际速度。"""
        low, high = self.lane_speed_ranges[lane_id]
        rng = self.road.np_random

        if not self.is_rule_breaker:
            self.target_speed = float(rng.uniform(low, high))

        elif rng.random() < 0.5:
            # 低速违规：低于当前车道下限。
            self.target_speed = float(
                rng.uniform(max(5.0, low - 8.0), low - 1.0)
            )

        else:
            # 超速违规：高于当前车道上限。
            self.target_speed = float(
                rng.uniform(high + 1.0, high + 8.0)
            )

    def _safe_target_lane(self, lane_index):
        """检查目标车道前后间距和正在汇入的车辆。"""
        lane = self.road.network.get_lane(lane_index)

        if not lane.is_reachable_from(self.position):
            return False

        front, rear = self.road.neighbour_vehicles(
            self, lane_index
        )

        for follower, leader in (
            (self, front),
            (rear, self),
        ):
            if follower is None or leader is None:
                continue

            gap = (
                float(leader.position[0] - follower.position[0])
                - (follower.LENGTH + leader.LENGTH) / 2
            )

            closing_speed = max(
                follower.speed - leader.speed, 0.0
            )

            required_gap = (
                10.0
                + 1.5 * max(follower.speed, 0.0)
                + 3.0 * closing_speed
            )

            if gap < required_gap:
                return False

        # 避免附近两辆车同时申请进入同一车道。
        for other in self.road.vehicles:
            if other is self:
                continue

            other_target = getattr(
                other, "target_lane_index", other.lane_index
            )

            if (
                other_target == lane_index
                and other.lane_index != lane_index
                and abs(other.position[0] - self.position[0]) < 80.0
            ):
                return False

        return True

    def _try_random_lane_change(self):
        """随机选择一个具备安全空间的相邻车道。"""
        if self.lane_index != self.target_lane_index:
            return

        # 当前车道归属可能在变道中途更新；
        # 等车身接近车道中心后再允许下一次变道。
        _, lateral = self.lane.local_coordinates(self.position)

        if abs(lateral) > 0.5:
            return

        candidates = list(
            self.road.network.side_lanes(self.lane_index)
        )
        self.road.np_random.shuffle(candidates)

        for candidate in candidates:
            if self._safe_target_lane(candidate):
                self.target_lane_index = candidate
                self.lane_change_request_count += 1
                break

    def act(self, action=None):
        if self.crashed:
            return

        current_lane_id = self.lane_index[2]

        # 所属车道变化时，适应新车道的速度范围。
        if current_lane_id != self.previous_lane_id:
            self._sample_target_speed(current_lane_id)
            self.previous_lane_id = current_lane_id

        if self.event_timer <= 0:
            # 新目标速度可能更高，也可能更低。
            self._sample_target_speed(current_lane_id)
            self.speed_event_count += 1

            if (
                self.road.np_random.random()
                < self.lane_change_probability
            ):
                self._try_random_lane_change()

            self.event_timer = self._next_interval()

        # 保留原 IDM 在变道过程中的冲突撤销检查。
        if self.lane_index != self.target_lane_index:
            super().change_lane_policy()

        # IDM负责跟车和加减速，底层控制器负责转向。
        super().act(action)

    def step(self, dt):
        self.event_timer -= dt
        super().step(dt)

    def acceleration(
        self,
        ego_vehicle,
        front_vehicle=None,
        rear_vehicle=None,
    ):
        """守规则车辆使用原IDM；违规车辆允许超速目标生效。"""
        if (
            ego_vehicle is not self
            or not self.is_rule_breaker
        ):
            return super().acceleration(
                ego_vehicle, front_vehicle, rear_vehicle
            )

        # 与当前版本IDM相同的计算方式，
        # 但对违规车辆不把目标速度截到车道限速。
        target = abs(utils.not_zero(self.target_speed))

        acceleration = self.COMFORT_ACC_MAX * (
            1.0
            - (max(self.speed, 0.0) / target) ** self.DELTA
        )

        if front_vehicle is not None:
            distance = self.lane_distance_to(front_vehicle)
            desired_gap = self.desired_gap(
                self, front_vehicle
            )

            acceleration -= self.COMFORT_ACC_MAX * (
                desired_gap / utils.not_zero(distance)
            ) ** 2

        return float(acceleration)







class TaskKinematicObservation(KinematicObservation):
    """车辆状态 + 剩余时间比例 + 剩余路程比例。"""

    def space(self):
        # 原来的车辆观测为17×5，共85个数
        vehicle_space = super().space()
        vehicle_size = int(np.prod(vehicle_space.shape))

        # 展平车辆观测，再增加两个任务变量
        return spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(vehicle_size + 3,),
            dtype=np.float32,
        )

    def observe(self):
        # 环境尚未创建车辆时，返回合法的空观测
        if self.env.vehicle is None:
            return np.zeros(self.space().shape, dtype=np.float32)

        vehicle_obs = super().observe().astype(np.float32).flatten()

        duration = float(self.env.config["duration"])
        target_distance = float(self.env.config["target_distance"])

        remaining_time_ratio = np.clip(
            (duration - self.env.time) / duration,
            0.0,
            1.0,
        )

        remaining_distance_ratio = np.clip(
            (
                target_distance
                - self.env.travelled_distance()
            ) / target_distance,
            0.0,
            1.0,
        )

        grace = self.env.config["slow_lane_grace_seconds"]

        slow_lane_ratio = float(np.clip(
            self.env.slow_lane_seconds / (2.0 * grace),
            0.0,
            1.0,
        ))

        task_obs = np.array(
            [
                remaining_time_ratio,
                remaining_distance_ratio,
                slow_lane_ratio,
            ],
            dtype=np.float32,
        )

        return np.concatenate([vehicle_obs, task_obs])








class ContinuousHighwayEnv(HighwayEnv):
    @classmethod
    def default_config(cls):
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics", "vehicles_count": 17,
                "features": ["presence", "x", "y", "vx", "vy"],
                "absolute": False, "normalize": True, "see_behind": True,
                "features_range": {
                    "x": [-400, 400], "y": [-16, 16],
                    "vx": [-50, 50], "vy": [-20, 20],
                },
            },
            "action": {
                "type": "DiscreteMetaAction",
                "target_speeds": [10, 15, 20, 25, 30, 35, 40, 45],
            },
            "lanes_count": 4, "duration": 60,
            "simulation_frequency": 15, "policy_frequency": 2,
            "initial_lane_id": 1, "initial_speed": 30.0,
            "preferred_speed": 30.0, "allowed_speed_range": [20.0, 40.0],
            "traffic_speed_range": [20.0, 40.0],
            # 每辆背景车生成时，成为速度违规车辆的概率。
            "traffic_violation_probability": 0.05,

            # 每次行为事件，尝试随机变道的概率。
            "traffic_lane_change_probability": 0.60,
            "traffic_event_interval": [3.0, 5.0],
            # 比车道最低速度低0.5m/s以上，才算明显低速。
            "slow_lane_speed_tolerance": 0.5,

            # 连续低速5秒，才开始处罚。
            "slow_lane_grace_seconds": 5.0,

            # 前车风险不超过0.05，才视为不需要明显避险。
            "slow_lane_front_risk_threshold": 0.05,
            "traffic_window": 400.0,
            "initial_offsets": [-300, -200, -100, 100, 200, 300],
            "initial_jitter": 5.0,
            "arrival_mean_seconds": 3.0,
            "spawn_band": [365.0, 395.0],
            "max_vehicles_per_lane": 8,
            "spawn_min_gap": 10.0, "safe_time_headway": 1.5,
            "critical_ttc": 3.0,
            "reward_weights": {
                "speed_tracking": 1.0,
                "speed_violation": 1.0,
                "front_risk": 2.0,
                "unsafe_lane_change": 2.0,

            # 显著提高失败成本，减少提前碰撞结束的收益
                "collision": 100.0,
                "offroad": 100.0,
                "slow_lane_stay": 1.0,
            },

            # 第一阶段暂时关闭赶进度奖励和每步时间成本
            "deadline_speed_weight": 0.0,
            "step_time_cost": 0.0,




            "normalize_reward": False, "offroad_terminal": True,
            "screen_width": 1000, "screen_height": 250,
            "scaling": 4.0, "centering_position": [0.5, 0.5],
            "lane_speed_ranges": [
                [35.0, 40.0],
                [30.0, 35.0],
                [25.0, 30.0],
                [20.0, 25.0],
            ],
            "target_distance": 1800.0,
            # "duration": 60,
            "success_bonus": 10.0,
            "timeout_penalty": 5.0,
            # "deadline_speed_weight": 0.5,
            # "step_time_cost": 0.05,
        })
        return config









    def define_spaces(self):
        # 先让 HighwayEnv 创建动作空间和原始观测空间
        super().define_spaces()

        observation_config = {
            key: value
            for key, value in self.config["observation"].items()
            if key != "type"
        }

        # 使用我们扩展的观测替换原观测
        self.observation_type = TaskKinematicObservation(
            self,
            **observation_config,
        )

        self.observation_space = self.observation_type.space()








    def required_average_speed(self):
        remaining_distance = max(
            0.0,
            self.config["target_distance"]
            - self.travelled_distance(),
        )

        # 避免时间为零时除零
        remaining_time = max(
            self.config["duration"] - self.time,
            0.5,
        )

        return remaining_distance / remaining_time



    def _reset(self):
        low, high = self.config["allowed_speed_range"]
        if not 0 < low < self.config["preferred_speed"] < high:
            raise ValueError("preferred_speed must be inside allowed_speed_range")
        band_low, band_high = self.config["spawn_band"]
        if not 0 < band_low < band_high < self.config["traffic_window"]:
            raise ValueError("spawn_band must be inside traffic_window")
        if self.config["arrival_mean_seconds"] <= 0:
            raise ValueError("arrival_mean_seconds must be positive")
        self.spawned = 0
        self.removed = 0

        # 每轮重新计时，并记录计时所属车道。
        self.slow_lane_seconds = 0.0
        self.slow_lane_id = None

        # super()._reset()
        super()._reset()
        self.start_x = float(self.vehicle.position[0])

    def travelled_distance(self):
        # 道路沿x轴，使用向前位移衡量完成路程
        return max(
            0.0,
            float(self.vehicle.position[0]) - self.start_x,
        )

    def reached_goal(self):
        return (
                self.travelled_distance()
                >= self.config["target_distance"]
        )

    def _is_terminated(self):
        # 碰撞、驶离道路或达到路程目标，结束本轮
        return super()._is_terminated() or self.reached_goal()





    def _create_road(self):
        # IDM clips desired speeds to the lane speed limit, so set it to 40.
        network = RoadNetwork.straight_road_network(
            lanes=self.config["lanes_count"], length=1_000_000,
            speed_limit=self.config["allowed_speed_range"][1],
        )
        for lane_id, speed_range in enumerate(
                self.config["lane_speed_ranges"]
        ):
            lane = network.get_lane(("0", "1", lane_id))
            lane.speed_limit = speed_range[1]
        self.road = Road(network=network, np_random=self.np_random)

    def _create_vehicles(self):
        lane_id = self.config["initial_lane_id"]
        lane = self.road.network.get_lane(("0", "1", lane_id))
        ego = self.action_type.vehicle_class(
            self.road, lane.position(1000, 0), heading=0,
            speed=self.config["initial_speed"],
        )
        self.controlled_vehicles = [ego]
        self.road.vehicles = [ego]
        for lane_id in range(self.config["lanes_count"]):
            for offset in self.config["initial_offsets"]:
                jitter = self.np_random.uniform(
                    -self.config["initial_jitter"], self.config["initial_jitter"]
                )
                placed = False
                for _ in range(100):
                    if self._spawn_vehicle(lane_id, ego.position[0] + offset + jitter):
                        placed = True
                        break
                if not placed:
                    raise ValueError("Initial offsets too close for safe traffic placement")

        # Independent arrival clocks on each lane and both window boundaries.
        self.next_arrival = {
            (lane_id, side): self._arrival_delay()
            for lane_id in range(self.config["lanes_count"])
            for side in (-1, 1)
        }

    def _arrival_delay(self):
        return max(0.1, float(self.np_random.exponential(
            self.config["arrival_mean_seconds"]
        )))

    def _safe_spawn_position(self, position, speed):
        # Check physical lateral overlap, including cars in a lane change.
        for other in self.road.vehicles:
            if abs(other.position[1] - position[1]) >= (other.WIDTH + IDMVehicle.WIDTH) / 2 + 0.5:
                continue
            delta = float(other.position[0] - position[0])
            follower_speed, leader_speed = (
                (speed, other.speed) if delta >= 0 else (other.speed, speed)
            )
            bumper_gap = abs(delta) - (other.LENGTH + IDMVehicle.LENGTH) / 2
            required = (
                self.config["spawn_min_gap"]
                + self.config["safe_time_headway"] * max(follower_speed, 0)
                + self.config["critical_ttc"] * max(follower_speed - leader_speed, 0)
            )
            if bumper_gap < required:
                return False
        return True

    def _spawn_vehicle(self, lane_id, longitudinal):
        lane = self.road.network.get_lane(("0", "1", lane_id))
        position = lane.position(longitudinal, 0)
        # speed = float(self.np_random.uniform(*self.config["traffic_speed_range"]))
        low, high = self.config["lane_speed_ranges"][lane_id]
        speed = float(self.np_random.uniform(low, high))
        if not self._safe_spawn_position(position, speed):
            return False
        vehicle = LaneSpeedIDMVehicle(
            self.road,
            position,
            heading=0,
            speed=speed,
            target_speed=speed,
            enable_lane_change=False,
            lane_speed_ranges=self.config["lane_speed_ranges"],
            violation_probability=(
                self.config["traffic_violation_probability"]
            ),
            lane_change_probability=(
                self.config["traffic_lane_change_probability"]
            ),
            event_interval=self.config["traffic_event_interval"],
        )
        vehicle.randomize_behavior()
        self.road.vehicles.append(vehicle)
        self.spawned += 1
        return True

    def _update_traffic(self):
        ego_x = self.vehicle.position[0]
        kept = [
            vehicle for vehicle in self.road.vehicles
            if vehicle is self.vehicle
            or abs(vehicle.position[0] - ego_x) <= self.config["traffic_window"]
        ]
        self.removed += len(self.road.vehicles) - len(kept)
        self.road.vehicles = kept
        for (lane_id, side), arrival in list(self.next_arrival.items()):
            if self.time < arrival:
                continue
            count = sum(
                v is not self.vehicle and v.lane_index[2] == lane_id
                for v in self.road.vehicles
            )
            if count < self.config["max_vehicles_per_lane"]:
                for _ in range(12):
                    offset = side * self.np_random.uniform(*self.config["spawn_band"])
                    if self._spawn_vehicle(lane_id, ego_x + offset):
                        break
            # Failed arrivals are skipped, not forced into an occupied gap.
            self.next_arrival[(lane_id, side)] = self.time + self._arrival_delay()

    def _simulate(self, action=None):
        start_time = self.time

        super()._simulate(action)

        # 使用实际仿真时间差，避免把决策步误当作秒。
        dt = max(0.0, self.time - start_time)
        self._update_slow_lane_timer(dt)

        if not self._is_terminated() and not self._is_truncated():
            self._update_traffic()

    def _pair_risk(self, follower, leader):
        distance = float(leader.position[0] - follower.position[0])
        gap = max(0.0, distance - (follower.LENGTH + leader.LENGTH) / 2)
        required = self.config["spawn_min_gap"] + self.config["safe_time_headway"] * max(follower.speed, 0)
        headway_risk = np.clip((required - gap) / required, 0, 1)
        closing_speed = follower.speed - leader.speed
        ttc = gap / closing_speed if closing_speed > 0.1 else np.inf
        ttc_risk = np.clip(1 - ttc / self.config["critical_ttc"], 0, 1)
        return float(max(headway_risk, ttc_risk))

    def _front_risk(self, lane_index):
        front, _ = self.road.neighbour_vehicles(self.vehicle, lane_index)
        return self._pair_risk(self.vehicle, front) if front is not None else 0.0




    def _update_slow_lane_timer(self, dt):
        """只累计前方风险低、稳定在本车道的明显低速时间。"""
        ego = self.vehicle
        lane_id = ego.lane_index[2]

        # 换了车道，旧车道的低速时间不再适用。
        if lane_id != self.slow_lane_id:
            self.slow_lane_seconds = 0.0
            self.slow_lane_id = lane_id

        low, _ = self.config["lane_speed_ranges"][lane_id]
        tolerance = self.config["slow_lane_speed_tolerance"]

        _, lateral = ego.lane.local_coordinates(ego.position)
        settled = (
            ego.lane_index == ego.target_lane_index
            and abs(lateral) <= 0.5
        )

        low_risk = (
            self._front_risk(ego.lane_index)
            <= self.config["slow_lane_front_risk_threshold"]
        )

        if (
            settled
            and low_risk
            and ego.speed < low - tolerance
        ):
            grace = self.config["slow_lane_grace_seconds"]

            # 计时上限为宽限时间的两倍。
            self.slow_lane_seconds = min(
                self.slow_lane_seconds + dt,
                2.0 * grace,
            )
        else:
            # 前方危险、恢复正常速度或开始变道：
            # 清零，重新给予减速宽限。
            self.slow_lane_seconds = 0.0

    def _safe_ego_lane(self, lane_index):
        """检查自车进入目标车道所需的前后空间。"""
        ego = self.vehicle
        lane = self.road.network.get_lane(lane_index)

        if not lane.is_reachable_from(ego.position):
            return False

        front, rear = self.road.neighbour_vehicles(
            ego, lane_index
        )

        for follower, leader in (
            (ego, front),
            (rear, ego),
        ):
            if follower is None or leader is None:
                continue

            gap = (
                float(leader.position[0] - follower.position[0])
                - (follower.LENGTH + leader.LENGTH) / 2
            )

            closing_speed = max(
                follower.speed - leader.speed, 0.0
            )

            required_gap = (
                self.config["spawn_min_gap"]
                + self.config["safe_time_headway"]
                * max(follower.speed, 0.0)
                + self.config["critical_ttc"]
                * closing_speed
            )

            if gap < required_gap:
                return False

        # 附近已有其他车辆汇入目标车道时，不鼓励自车跟着汇入。
        for other in self.road.vehicles:
            if other is ego:
                continue

            other_target = getattr(
                other, "target_lane_index", other.lane_index
            )

            if (
                other_target == lane_index
                and other.lane_index != lane_index
                and abs(other.position[0] - ego.position[0]) < 80.0
            ):
                return False

        return True

    def _slow_lane_stay_penalty(self):
        """有合适且安全的右侧车道时，处罚长期低速占道。"""
        ego = self.vehicle
        lane_id = ego.lane_index[2]
        grace = self.config["slow_lane_grace_seconds"]

        if (
            self.slow_lane_seconds <= grace
            or ego.crashed
            or not ego.on_road
            or ego.lane_index != ego.target_lane_index
        ):
            return 0.0

        if (
            self._front_risk(ego.lane_index)
            > self.config["slow_lane_front_risk_threshold"]
        ):
            return 0.0

        # 已在最右侧，就没有更慢的右侧车道。
        right_id = lane_id + 1
        if right_id >= self.config["lanes_count"]:
            return 0.0

        current_low, _ = self.config["lane_speed_ranges"][lane_id]
        right_low, right_high = (
            self.config["lane_speed_ranges"][right_id]
        )
        tolerance = self.config["slow_lane_speed_tolerance"]

        if ego.speed >= current_low - tolerance:
            return 0.0

        # 右侧车道必须与当前速度匹配。
        if not (
            right_low - tolerance
            <= ego.speed
            <= right_high + tolerance
        ):
            return 0.0

        right_index = (
            ego.lane_index[0],
            ego.lane_index[1],
            right_id,
        )

        if not self._safe_ego_lane(right_index):
            return 0.0

        # 第5秒开始处罚，第10秒达到最大处罚1。
        return float(np.clip(
            (self.slow_lane_seconds - grace) / grace,
            0.0,
            1.0,
        ))

    def _rewards(self, action):
        ego = self.vehicle
        current_risk = self._front_risk(ego.lane_index)
        target_risk = self._front_risk(ego.target_lane_index)
        front_risk = max(current_risk, target_risk)
        changing_lane = ego.lane_index != ego.target_lane_index
        _, rear = self.road.neighbour_vehicles(ego, ego.target_lane_index)
        rear_risk = self._pair_risk(rear, ego) if rear is not None else 0.0

        speed = max(0.0, float(ego.speed))
        lane_id = ego.lane_index[2]

        low, high = self.config["lane_speed_ranges"][lane_id]
        # 正常情况下，偏好当前车道的速度中点
        lane_preferred = (low + high) / 2

        # 时间紧、进度落后时，提高偏好速度
        required_speed = self.required_average_speed()

        # 偏好速度始终限制在当前车道允许范围内
        preferred = float(np.clip(
            max(lane_preferred, required_speed),
            low,
            high,
        ))

        deviation = abs(speed - preferred) / preferred




        # Relax slow-speed costs only when a nearby front vehicle poses risk.
        if speed < preferred:
            deviation *= 1 - front_risk
        # tracking = 1 - min(deviation, 1.0)
        # 匹配目标速度时为0，偏离时为负
        # 减少长期停留在环境中获取正奖励的动机
        tracking = -min(deviation, 1.0)
        violation = (
            max(low - speed, 0) / low * (1 - front_risk)
            + max(speed - high, 0) / high
        )
        return {
            "speed_tracking": tracking,
            "speed_violation": violation,
            "front_risk": front_risk,
            "unsafe_lane_change": rear_risk if changing_lane else 0.0,
            "collision": float(ego.crashed),
            "offroad": float(not ego.on_road),
            "slow_lane_stay": self._slow_lane_stay_penalty(),
        }

    def _reward(self, action):
        components = self._rewards(action)
        weights = self.config["reward_weights"]

        reward = (
                weights["speed_tracking"]
                * components["speed_tracking"]
                - sum(
            weights[key] * value
            for key, value in components.items()
            if key != "speed_tracking"
        )
        )


        # 每一步付出少量时间成本，鼓励及时完成任务
        reward -= self.config["step_time_cost"]

        required_speed = self.required_average_speed()
        speed = max(0.0, float(self.vehicle.speed))

        # 所需速度超过实际速度的程度
        speed_shortfall = float(np.clip(
            (required_speed - speed)
            / self.config["allowed_speed_range"][1],
            0.0,
            1.0,
        ))

        # 越接近截止时间，进度不足的处罚越强
        urgency = float(np.clip(
            self.time / self.config["duration"],
            0.0,
            1.0,
        ))

        # 前方危险时，减轻赶进度的压力
        safety_relief = 1.0 - components["front_risk"]

        reward -= (
            self.config["deadline_speed_weight"]
            * urgency
            * speed_shortfall
            * safety_relief
        )




        safe = (
                not self.vehicle.crashed
                and self.vehicle.on_road
        )

        if self.reached_goal() and safe:
            reward += self.config["success_bonus"]

        elif (
                self.time >= self.config["duration"]
                and not self.reached_goal()
        ):
            reward -= self.config["timeout_penalty"]

        return float(reward)

    def _info(self, obs, action=None):
        info = super()._info(obs, action)
        deltas = [v.position[0] - self.vehicle.position[0]
                  for v in self.road.vehicles if v is not self.vehicle]
        lane_id = self.vehicle.lane_index[2]
        low, high = self.config["lane_speed_ranges"][lane_id]
        info.update({

            "remaining_time": max(
                0.0,
                self.config["duration"] - self.time,
            ),
            "required_average_speed": self.required_average_speed(),

            "traffic_front": sum(d > 0 for d in deltas),
            "traffic_rear": sum(d < 0 for d in deltas),
            "traffic_spawned": self.spawned,
            "traffic_removed": self.removed,
            "speed_violation": not low <= self.vehicle.speed <= high,
            "distance": self.travelled_distance(),
            "success": (
                    self.reached_goal()
                    and not self.vehicle.crashed
                    and self.vehicle.on_road
            ),
            "background_rule_breakers": sum(
                bool(getattr(v, "is_rule_breaker", False))
                for v in self.road.vehicles
                if v is not self.vehicle
            ),

            "background_speed_events": sum(
                getattr(v, "speed_event_count", 0)
                for v in self.road.vehicles
                if v is not self.vehicle
            ),

            "background_lane_change_requests": sum(
                getattr(v, "lane_change_request_count", 0)
                for v in self.road.vehicles
                if v is not self.vehicle
            ),
            "slow_lane_seconds": float(self.slow_lane_seconds),
            "slow_lane_stay_penalty": self._slow_lane_stay_penalty(),
        })
        return info
