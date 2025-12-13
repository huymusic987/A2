# ============================================================================
# frog.py
# Purpose
#   Player controlled agent. Moves with Arrive.
# Update order
#   Compute steering, integrate velocity with dt, clamp to bounds, update bubbles.
# Drawing
#   Draw the frog body and a simple eye that points in the facing direction.
# ============================================================================

import pygame
from pygame.math import Vector2 as V2
from steering import seek, arrive, integrate_velocity
from constants import WHITE, GREEN, FROG_RADIUS, FROG_SPEED, GRID_WIDTH, GRID_HEIGHT

def clamp(x, a, b):
    """Limit a scalar value x so it stays between a and b inclusive."""
    return max(a, min(b, x))

class Frog:
    def __init__(self, pos):
        self.pos = V2(pos)
        self.vel = V2()
        self.target = V2(pos)
        self.radius = FROG_RADIUS
        self.speed = FROG_SPEED
        self.facing = V2(1, 0)

        self.path = []
        self.path_index = 0
        self.waypoint_radius = 10

    def set_target(self, p):
        """Set a new target the frog will move toward using Arrive."""
        self.target = V2(p)

    def set_path(self, path):
        """Receive A* path as list of (row, col) grid cells."""
        cell_size = 40 
        self.path = []
        for (r, c) in path:
            px = c * cell_size + cell_size / 2
            py = r * cell_size + cell_size / 2
            self.path.append(V2(px, py))

        self.path_index = 0

        # If empty path, do nothing
        if not self.path:
            return

        # First target = first waypoint center
        self.target = self.path[0]


    def update(self, dt):
        # ---------------------------------------------------------
        # 1. If NO PATH → do nothing (stay still)
        # ---------------------------------------------------------
        if not self.path or self.path_index >= len(self.path):
            self.vel = V2()   # ensure fully stopped
            return

        # ---------------------------------------------------------
        # 2. PATH FOLLOWING → get current waypoint
        # ---------------------------------------------------------
        waypoint = self.path[self.path_index]

        # If close enough, advance to next waypoint
        if (self.pos - waypoint).length() < self.waypoint_radius:
            self.path_index += 1

            # If path finished, stop completely
            if self.path_index >= len(self.path):
                self.vel = V2()
                return

            waypoint = self.path[self.path_index]

        # ---------------------------------------------------------
        # 3. Detect if we're approaching a turn
        # ---------------------------------------------------------
        is_turning = False
        steer_multiplier = 1.0
    
        if self.path_index < len(self.path) - 1:
            dist_to_waypoint = (self.pos - waypoint).length()
            turn_detection_distance = 75  # Detect turns within this distance
        
            if dist_to_waypoint < turn_detection_distance:
                is_turning = True
                # Boost steering力 during turns for sharper response
                steer_multiplier = 2.5  # Increase for even sharper turns

        # ---------------------------------------------------------
        # 4. Steering
        # ---------------------------------------------------------
        if self.path_index == len(self.path) - 2:
            steer = arrive(self.pos, self.vel, waypoint, self.speed)
        else:
            steer = seek(self.pos, self.vel, waypoint, self.speed)
    
        # Apply stronger steering during turns
        if is_turning:
            steer = steer * steer_multiplier

        self.target = waypoint

        # ---------------------------------------------------------
        # 5. Integrate velocity + move frog
        # ---------------------------------------------------------
        self.vel = integrate_velocity(self.vel, steer, dt, self.speed)
        self.pos += self.vel * dt

        # Face in movement direction
        if self.vel.length_squared() > 16:
            self.facing = self.vel.normalize()

        # Clamp to world bounds
        self.pos.x = clamp(self.pos.x, self.radius, GRID_WIDTH - self.radius)
        self.pos.y = clamp(self.pos.y, self.radius, GRID_HEIGHT - self.radius)


    def draw(self, surf):
        color = GREEN

        # Body
        pygame.draw.circle(surf, color, self.pos, self.radius)

        # Eye looks in facing direction
        eye = self.pos + self.facing * (self.radius - 4)
        pygame.draw.circle(surf, WHITE, eye, 5)
        pygame.draw.circle(surf, (25, 25, 25), eye, 2)
