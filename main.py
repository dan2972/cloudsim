import math

from camera import Camera
import numpy as np
import slangpy as spy

class Engine:
    WIDTH = 1280
    HEIGHT = 720

    def __init__(self):
        self.window = spy.Window(width=self.WIDTH, height=self.HEIGHT, title="Engine")
        # self.device = spy.create_device(type=spy.DeviceType.vulkan)
        self.device = spy.create_device() # auto select backend
        print(self.device)
        self.surface = self.device.create_surface(self.window)
        self.surface.configure(width=self.WIDTH, height=self.HEIGHT, vsync=False)

        self.output_texture = None

        self.render_program = self.device.load_program('tracer', ['render'])
        self.render_kernel = self.device.create_compute_kernel(self.render_program)

        self.ui_context = spy.ui.Context(self.device)
        self.setup_ui()

        self.fps_avg = 0.0

        self.window.on_keyboard_event = self.on_keyboard_event
        self.window.on_mouse_event = self.on_mouse_event

        self.camera = Camera(position=np.array([0.0, 10.0, -5.0], dtype=np.float32))

        self.pressed_keys = set()
        self.mouse_pos = spy.float2()

        self.capture_mouse = False

        self.volume_texture = self.load_volume_data('cloud_4.npy', self.device)
        self.volume_sampler = self.device.create_sampler(
            min_filter=spy.TextureFilteringMode.linear,
            mag_filter=spy.TextureFilteringMode.linear,
            mip_filter=spy.TextureFilteringMode.linear,
            address_u=spy.TextureAddressingMode.clamp_to_border,
            address_v=spy.TextureAddressingMode.clamp_to_border,
            address_w=spy.TextureAddressingMode.clamp_to_border,
            border_color=np.array([0,0,0,0], dtype=np.float32)
        )

    def run(self):
        frame = 0
        time = 0.0
        timer = spy.Timer()

        while not self.window.should_close():
            self.window.process_events()

            elapsed = timer.elapsed_s()
            timer.reset()

            self.handle_key_input(elapsed)

            time += elapsed

            self.fps_avg = 0.95 * self.fps_avg + 0.05 * (1.0 / elapsed)
            self.fps_text.text = f'FPS: {self.fps_avg:.2f}'

            surface_texture = self.surface.acquire_next_image()
            if not surface_texture:
                continue

            self.ui_context.begin_frame(surface_texture.width, surface_texture.height)

            if self.output_texture is None or \
                self.output_texture.width != self.WIDTH or \
                self.output_texture.height != self.HEIGHT:
                self.output_texture = self.device.create_texture(
                    format=spy.Format.rgba16_float,
                    width=surface_texture.width, 
                    height=surface_texture.height,
                    usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
                    label="output_texture",
                )

            command_encoder = self.device.create_command_encoder()
            self.render_kernel.dispatch(
                thread_count=[surface_texture.width, surface_texture.height, 1],
                vars={
                    'camera': self.camera.get_data(),
                    'resolution': np.array([float(self.WIDTH), float(self.HEIGHT)], dtype=np.float32),
                    'output': self.output_texture,
                    'volume': self.volume_texture,
                    'volumeSampler': self.volume_sampler,
                    'sunDirection': get_sun_direction(self.slider1.value),
                    'cloudScatterCoeff': self.slider2.value,
                    'densityScale': self.slider3.value,
                    'ambientIntensity': self.slider4.value,
                    'sunIntensity': self.slider5.value,
                },
                command_encoder=command_encoder
            )
            command_encoder.blit(surface_texture, self.output_texture)

            self.ui_context.end_frame(surface_texture, command_encoder)

            self.device.submit_command_buffer(command_encoder.finish())
            del surface_texture

            self.surface.present()

            frame += 1

    def setup_ui(self):
        screen = self.ui_context.screen
        window = spy.ui.Window(screen, "Settings", size=spy.float2(400, 100))

        self.fps_text = spy.ui.Text(window, 'FPS: 0')

        self.slider1 = spy.ui.SliderFloat(window, 'time', value=0.5, min=-0.1, max=1.1)
        self.slider2 = spy.ui.SliderFloat(window, 'cloud scatter coeff', value=0.9, min=0.0, max=1.0)
        self.slider3 = spy.ui.SliderFloat(window, 'density', value=1.0, min=0.1, max=1.0)
        self.slider4 = spy.ui.SliderFloat(window, 'ambient', value=0.05, min=0.0, max=1.0)
        self.slider5 = spy.ui.SliderFloat(window, 'sunIntensity', value=50.0, min=0, max=100.0)

    def on_keyboard_event(self, event: spy.KeyboardEvent):
        if event.type == spy.KeyboardEventType.key_press:
            self.pressed_keys.add(event.key)
            if event.key == spy.KeyCode.escape:
                self.window.close()
        elif event.type == spy.KeyboardEventType.key_release:
            if event.key in self.pressed_keys:
                self.pressed_keys.remove(event.key)

    def handle_key_input(self, dt):
        speed = 5 * dt

        if spy.KeyCode.left_control in self.pressed_keys:
            speed *= 5

        if spy.KeyCode.w in self.pressed_keys:
            self.camera.move('forward', speed)
        if spy.KeyCode.s in self.pressed_keys:
            self.camera.move('backward', speed)
        if spy.KeyCode.a in self.pressed_keys:
            self.camera.move('left', speed)
        if spy.KeyCode.d in self.pressed_keys:
            self.camera.move('right', speed)
        if spy.KeyCode.space in self.pressed_keys:
            self.camera.move('up', speed)
        if spy.KeyCode.left_shift in self.pressed_keys:
            self.camera.move('down', speed)

    def on_mouse_event(self, event: spy.MouseEvent):
        if self.ui_context.handle_mouse_event(event):
            return
        
        if event.type == spy.MouseEventType.move:
            if self.capture_mouse:
                delta = event.pos - self.mouse_pos
                self.camera.rotate(delta.x, delta.y)
            self.mouse_pos = event.pos
        if event.type == spy.MouseEventType.button_down:
            self.capture_mouse = not self.capture_mouse
            if self.capture_mouse:
                self.window.cursor_mode = spy.CursorMode.disabled
            else:
                self.window.cursor_mode = spy.CursorMode.normal
    
    def load_volume_data(self, path, device):
        density_data = np.load(path)
        d, h, w = density_data.shape
        volume_texture = device.create_texture(
            type=spy.TextureType.texture_3d,
            format=spy.Format.r16_float,
            width=w,
            height=h,
            depth=d,
            usage=spy.TextureUsage.shader_resource,
            label="volume_texture",
            data=density_data.astype(np.float16)
        )
        return volume_texture

def get_sun_direction(time):
    # 0 = Sunrise (East), PI/2 = Noon (Top), PI = Sunset (West)
    angle = time * math.pi
    x = math.cos(angle)
    y = math.sin(angle)
    z = 0.0
    return spy.float3(x, y, z)

if __name__ == "__main__":
    engine = Engine()
    engine.run()