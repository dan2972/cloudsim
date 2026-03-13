import numpy as np

class Camera:
    def __init__(self, position, rotation=np.array([np.pi/2, 0.0]), sensitivity=0.002):
        self.position = position
        self.rotation = rotation
        self.sensitivity = sensitivity
    
    def get_camera_vectors(self):
        yaw = self.rotation[0]
        pitch = np.clip(self.rotation[1], -np.deg2rad(89), np.deg2rad(89))

        forward = np.array([
            np.cos(yaw) * np.cos(pitch),
            np.sin(pitch),
            np.sin(yaw) * np.cos(pitch)
        ], dtype=np.float32)
        
        world_up = np.array([0, 1, 0], dtype=np.float32)
        right = np.cross(forward, world_up)
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        
        return forward, right, up
    
    def move(self, direction, amount):
        forward, right, up = self.get_camera_vectors()
        world_up = np.array([0, 1, 0], dtype=np.float32)
        forward = np.array([forward[0], 0, forward[2]], dtype=np.float32)
        forward /= np.linalg.norm(forward)
        
        if direction == 'forward':
            self.position += forward* amount
        elif direction == 'backward':
            self.position -= forward* amount
        elif direction == 'left':
            self.position -= right* amount
        elif direction == 'right':
            self.position += right* amount
        elif direction == 'up':
            self.position += world_up * amount
        elif direction == 'down':
            self.position -= world_up * amount
    
    def rotate(self, dx, dy):
        self.rotation[0] += dx * self.sensitivity
        self.rotation[1] -= dy * self.sensitivity
    
    def get_data(self):
        forward, right, up = self.get_camera_vectors()
        return {
            'position': self.position,
            'forward': forward,
            'right': right,
            'up': up
        }