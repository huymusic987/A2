# ============================================================================
# steering.py
# Purpose
#   Implement all steering behaviours here. Each function computes a steering
#   force vector. Entities apply that force to their velocity each frame.
# Key idea
#   desired_velocity minus current_velocity gives the steering force.
#   Use dt in update loops when integrating velocity to keep motion consistent.
# ============================================================================

from pygame.math import Vector2 as V2

# ---------------- Base behaviours ----------------

def limit(v, max_len):
    """
    Limit a vector length.
    If v is longer than max_len, scale it down to exactly max_len.
    """
    if v.length_squared() > max_len * max_len:
        v.scale_to_length(max_len)
    return v

def integrate_velocity(vel, force, dt, max_speed):
    """
    Apply a steering force to velocity using Euler integration.
    Then clamp to max speed and return the new velocity.
    Use this inside agent update methods after computing steering forces.
    """
    vel += limit(force, 500.0) * dt
    if vel.length() > max_speed:
        vel.scale_to_length(max_speed)
    return vel

def seek(pos, vel, target, max_speed):
    """
    Move toward a target. Returns a steering force.
    desired = direction_to_target * max_speed
    steering = desired - current_velocity
    """
    d = target - pos
    if d.length_squared() == 0:
        return V2()
    desired = d.normalize() * max_speed
    return desired - vel

def arrive(pos, vel, target, max_speed, slow_radius=100, stop_radius=-20):
    """
    Like seek when far, but slow down near the target.
    Rules
      If distance < stop_radius, return a force that cancels leftover velocity
      If distance < slow_radius, scale desired speed by distance / slow_radius
      Otherwise use full speed
    This should remove overshoot and jitter around the target.
    """
    displacement_vector = target - pos

    distance = displacement_vector.length()

    if distance < stop_radius:
        return -vel

    if distance < slow_radius:
        scaled_speed = max_speed * (distance / slow_radius) 
    else:
        scaled_speed = max_speed
    
    desired_velocity = displacement_vector.normalize() * scaled_speed

    return desired_velocity - vel

# def path_following(pos, vel, path, current_idx, path_radius, max_speed):
#     if not path or current_idx >= len(path):
#         return V2(), current_idx
    
#     closest_dist = float('inf')
#     closest_idx = current_idx
#     closest_point = path[current_idx]
    
#     for i in range(current_idx, min(current_idx + 3, len(path))):
#         waypoint = path[i]
#         dist = (waypoint - pos).length()
#         if dist < closest_dist:
#             closest_dist = dist
#             closest_idx = i
#             closest_point = waypoint
            
#     if closest_idx > current_idx:
#         current_idx = closest_idx
#     elif closest_dist < path_radius and current_idx < len(path) - 1:
#         current_idx += 1
    
#     lookahead_idx = min(current_idx + 1, len(path) - 1)
#     target_point = path[lookahead_idx]
    
#     is_arrive_zone = (lookahead_idx >= len(path) - 4)
    
#     if is_arrive_zone:
#         steer = arrive(pos, vel, target_point, max_speed)
#     else:
#         steer = seek(pos, vel, target_point, max_speed)
    
#     if closest_dist > path_radius * 0.3:
#         correction = (closest_point - pos).normalize() * max_speed * 0.5
#         steer += correction
    
#     return steer, current_idx

# # ---------------- Boids components ----------------

# def boids_separation(me_pos, neighbors, sep_radius):
#     """
#     Push away from neighbors that are too close.
#     neighbors: list of tuples (neighbor_pos, neighbor_vel)
#     Typical approach
#       For each neighbor inside sep_radius, add a vector pointing away with
#       magnitude inversely proportional to distance. Normalize at the end.
#     """
#     if not neighbors:
#         return V2()
    
#     separation_force = V2()
#     for neighbor_position, _ in neighbors:
#         offset_vector = me_pos - neighbor_position
#         distance = offset_vector.length()

#         if 0 < distance < sep_radius:
#             separation_force += offset_vector.normalize() / distance

#     if separation_force.length_squared() > 0:
#         separation_force = separation_force.normalize()

#     return separation_force

# def boids_cohesion(me_pos, neighbors):
#     """
#     Pull toward the average position of neighbors.
#     Typical approach
#       Compute the center of mass of neighbors then steer toward that point.
#     """
#     if not neighbors:
#         return V2()

#     total_position = V2()
#     for neighbor_position, _ in neighbors:
#         total_position += neighbor_position

#     average_neighbor_position = total_position / len(neighbors)
#     direction_to_center = average_neighbor_position - me_pos

#     if direction_to_center.length_squared() > 0:
#         return direction_to_center.normalize()
#     return V2()

# def boids_alignment(me_vel, neighbors):
#     """
#     Match the average velocity of neighbors.
#     Typical approach
#       Compute the average heading of neighbors then steer toward that heading.
#     """
#     if not neighbors:
#         return V2()

#     total_velocity = V2()
#     for _, neighbor_velocity in neighbors:
#         total_velocity += neighbor_velocity

#     average_velocity = total_velocity / len(neighbors)
#     desired_direction = (
#         average_velocity.normalize()
#         if average_velocity.length_squared() > 0
#         else V2()
#     )

#     if me_vel.length_squared() > 0:
#         return desired_direction - me_vel.normalize()
#     return desired_direction

# # ---------------- Obstacle avoidance blend ----------------

# def seek_with_avoid(pos, vel, target, max_speed, radius, rects, lookahead=AVOID_LOOKAHEAD):
#     """
#     Seek the target but avoid obstacles by sampling angled corridors.
#     Idea
#       1. Check a straight corridor first
#       2. If blocked, rotate small angles left and right until a free path is found
#       3. Use that direction for the seek
#       4. If all blocked, apply a small braking force
#     Use circlecast_hits_any_rect to test each corridor.
#     """
#     displacement_vector = target - pos
#     if displacement_vector.length_squared() == 0:
#         return V2()

#     forward_direction = displacement_vector.normalize()

#     straight_probe = pos + forward_direction * lookahead
#     if not circlecast_hits_any_rect(pos, straight_probe, radius, rects):
#         desired_velocity = forward_direction * max_speed
#         return desired_velocity - vel

#     for angle in range(AVOID_ANGLE_INCREMENT, AVOID_MAX_ANGLE + 1, AVOID_ANGLE_INCREMENT):
#         for direction_sign in (+1, -1):
#             rotated_direction = forward_direction.rotate(direction_sign * angle)
#             probe_point = pos + rotated_direction * lookahead

#             if not circlecast_hits_any_rect(pos, probe_point, radius, rects):
#                 desired_velocity = rotated_direction * max_speed
#                 return desired_velocity - vel

#     return -vel * 0.5

# # ---------------- New behaviours to be implemented ----------------

# def pursue(pos, vel, target_pos, target_vel, max_speed):
#     """
#     Predict the future position of the target then seek that point.
#     Suggested
#       distance = |target_pos - pos|
#       time_horizon = distance / (max_speed + small_eps)
#       predicted    = target_pos + target_vel * time_horizon
#       return seek toward predicted
#     Replace simple seek in Snake Aggro with pursue for better interception.
#     """
#     displacement_vector = target_pos - pos
#     distance_to_target = displacement_vector.length()
#     if distance_to_target == 0:
#         return V2()
    
#     time_horizon = distance_to_target / (max_speed + 1e-5)
#     predicted_target_position = target_pos + target_vel * time_horizon
    
#     displacement_vector_from_predicted = predicted_target_position - pos
#     if displacement_vector_from_predicted.length_squared() > 0:
#         desired_velocity = displacement_vector_from_predicted.normalize() * max_speed
#     else:
#         desired_velocity = V2()
    
#     return desired_velocity - vel

# def evade(pos, vel, threat_pos, threat_vel, max_speed):
#     """
#     Predict the future position of a threat then flee from that point.
#     This is the inverse of pursue. Use the same prediction idea.
#     """
#     displacement_vector = threat_pos - pos
#     distance_to_threat = displacement_vector.length()
#     if distance_to_threat == 0:
#         return V2()
    
#     time_horizon = distance_to_threat / (max_speed + 1e-5)
#     predicted_threat_position = threat_pos + threat_vel * time_horizon
    
#     displacement_vector_from_predicted = pos - predicted_threat_position
#     if displacement_vector_from_predicted.length_squared() > 0:
#         desired_velocity = displacement_vector_from_predicted.normalize() * max_speed
#     else:
#         desired_velocity = V2()
    
#     return desired_velocity - vel

# def wander_force(me_vel, jitter_deg=12.0, circle_distance=24.0, circle_radius=18.0, rng_seed=None):
#     """
#     Return a small random steering vector for gentle drift.
#     Classic wander
#       Project a small circle ahead along current heading, then jitter the
#       target point on that circle by a tiny random angle each update.
#     Use this for Fly Idle and Snake Confused.
#     """
#     rng = random.Random(rng_seed) if rng_seed is not None else random

#     if me_vel.length_squared() > 0:
#         heading = me_vel.normalize()
#     else:
#         angle = rng.uniform(0, 2 * math.pi)
#         heading = V2(math.cos(angle), math.sin(angle))
    
#     wander_theta = rng.uniform(0, 2 * math.pi)

#     jitter_rad = math.radians(jitter_deg)
#     wander_theta += rng.uniform(-jitter_rad, jitter_rad)

#     circle_center = heading * circle_distance

#     displacement = V2(
#         math.cos(wander_theta) * circle_radius,
#         math.sin(wander_theta) * circle_radius
#     )

#     wander_target = circle_center + displacement

#     if wander_target.length_squared() > 0:
#         return wander_target.normalize()

#     return V2()
