import pygame
import numpy as np
import slangpy as spy

# Initialize Pygame
WIDTH, HEIGHT = 800, 600
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

# Initialize SlangPy
device = spy.create_device()
module = spy.Module.load_from_file(device, "tracer.slang")

# Camera state
cam_pos = np.array([0.0, 0.0, -5.0], dtype=np.float32)
cam_rot = 0.0

def get_camera_vectors(rot):
    forward = np.array([np.sin(rot), 0, np.cos(rot)], dtype=np.float32)
    right = np.array([np.cos(rot), 0, -np.sin(rot)], dtype=np.float32)
    up = np.array([0, 1, 0], dtype=np.float32)
    return forward, right, up

# Create GPU Texture to render into
out_texture = device.create_texture(
    type=spy.TextureType.texture_2d,
    format=spy.Format.rgba32_float,
    width=WIDTH, height=HEIGHT,
    usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access
)

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Movement Logic
    keys = pygame.key.get_pressed()
    speed = 0.1
    forward, right, up = get_camera_vectors(cam_rot)
    
    if keys[pygame.K_w]: cam_pos += forward * speed
    if keys[pygame.K_s]: cam_pos -= forward * speed
    if keys[pygame.K_a]: cam_pos -= right * speed
    if keys[pygame.K_d]: cam_pos += right * speed
    if keys[pygame.K_LEFT]: cam_rot -= 0.05
    if keys[pygame.K_RIGHT]: cam_rot += 0.05

    # Dispatch Shader
    camera_data = {
        'position': cam_pos,
        'forward': forward,
        'right': right,
        'up': up
    }
    

    module.render(
        tid=spy.grid(shape=(HEIGHT, WIDTH)),
        camera=camera_data,
        resolution=[float(WIDTH), float(HEIGHT)],
        _result=out_texture
    )

    # Copy GPU texture to CPU for display
    # (Note: In a high-perf app, you'd use a shared buffer, but this is simple)
    frame_data = out_texture.to_numpy()
    frame_data = (np.clip(frame_data, 0, 1) * 255).astype(np.uint8)
    
    # Render to Pygame window
    surface = pygame.surfarray.make_surface(frame_data[:, :, :3].transpose(1, 0, 2))
    screen.blit(surface, (0, 0))
    pygame.display.flip()
    
    pygame.display.set_caption(f"SlangPy Raytracer - FPS: {clock.get_fps():.1f}")
    clock.tick(60)

pygame.quit()