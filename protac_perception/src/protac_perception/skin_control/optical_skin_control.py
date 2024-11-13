import serial
from typing import Literal, Type
import time
import threading
from protac_perception.util.cam_control import CameraControl


class PDLCControl:
    def __init__(self,
                 serial_port: str,
                 pdlc_type: Literal["normal", "reverse"]):
        # Open the serial connection
        self.serial = serial.Serial(port=serial_port,
                                    baudrate=115200,
                                    bytesize=serial.EIGHTBITS,
                                    parity=serial.PARITY_NONE,
                                    stopbits=serial.STOPBITS_ONE,
                                    xonxoff=False,
                                    rtscts=True,
                                    timeout=1)
        self.pdlc_type = pdlc_type

    def control_pdlc(self, mode: Literal["opaque", "transparent"]):
        """
        Switching pdlc-film transparency back and forth
        pdlc_type == "normal" | "reverse"
        mode == "opaque" | "transparent" 
        """
        # Define the message values
        start = 2
        on_data = '8104040009' # ON voltage: 40V
        # on_data = '8104005008' # ON voltage: 5V 
        off_data = '800008'
        end = 3
        
        # Check if the connection is open
        if self.serial.is_open:
            if self.pdlc_type == "normal":
                if mode == "transparent":
                    message = f'{chr(start)}{on_data}{chr(end)}'
                elif mode == "opaque":
                    message = f'{chr(start)}{off_data}{chr(end)}'
                else:
                    print("Invalid PDLC mode."
                          "Please choose transparent | opaque")
                # Convert the message string to bytes
                message_bytes = message.encode()
                # Send the message
                self.serial.write(message_bytes)
                print(mode)
                                
                # # Read a response
                # response = self.serial.readline()
                # response_str = response.decode()
                # print("Response received: {}".format(response_str))

            elif self.pdlc_type == "reverse":
                if mode == "transparent":
                    message = f'{chr(start)}{off_data}{chr(end)}'
                    # Convert the message string to bytes
                    message_bytes = message.encode()
                elif mode == "opaque":
                    message = f'{chr(start)}{on_data}{chr(end)}'
                else:
                    print("Invalid PDLC mode."
                          "Please choose transparent | opaque")
                # Convert the message string to bytes
                message_bytes = message.encode()
                # Send the message
                self.serial.write(message_bytes)
                # # Read a response
                # response = self.serial.readline()
                # response_str = response.decode()
                # print("Response received: {}".format(response_str))
            else:
                print("Invalid PDLC type")
        else:
            print("Fail to open the serial port."
                  "Please check the connection!")
            
    def close_serial_port(self):
        self.serial.close()


class OpticalSkinControl:
    def __init__(self, 
                 cam: Type[CameraControl], 
                 pdlc_serial_port: str,
                 led_serial_port: str,
                 pdlc_type: Literal["normal", "reverse"]):
        
        self.pdlc_control = PDLCControl(pdlc_serial_port, pdlc_type)
        self.led_control = LEDControl(serial_port=led_serial_port)
        self.cam = cam
        self.os = self.cam.os
        self.set_opaque()
        self.skin_state = "opaque" # "transparent" | "opaque"
        
    def set_transparent(self):
        self.skin_state = "transparent"
        self.switch_time = time.time()
        self.cam.set_image_perspective(False)
        self.pdlc_control.control_pdlc("transparent")
        self.led_control.led_off()
        if self.os == 'windows':
            self.cam.set_exposure(-5)
        elif self.os == 'ubuntu':
            self.cam.set_exposure(157)
        self.cam.set_brightness(0)
        self.cam.set_contrast(32)
        
    def set_opaque(self):
        self.skin_state = "opaque"
        self.switch_time = time.time()
        self.cam.set_image_perspective(False)
        self.pdlc_control.control_pdlc("opaque")
        self.led_control.led_on()
        if self.os == 'windows':
            self.cam.set_exposure(-7)
        elif self.os == 'ubuntu':
            self.cam.set_exposure(100)
        self.cam.set_brightness(-30)
        self.cam.set_contrast(64)
        
    def terminate_pdlc_control(self):
        self.pdlc_control.close_serial_port()
        self.led_control.close_serial_port()

    def switch_state_periodically(self):
        self.running = True
        while self.running:
            with self.lock:
                if self.skin_state == "transparent":
                    self.set_opaque()
                else:
                    self.set_transparent()
            time.sleep(self.dt)

    def start_switching(self, dt):
        self.lock = threading.Lock()
        self.dt = dt
        # Start the switching thread
        switch_thread = threading.Thread(target=self.switch_state_periodically)
        switch_thread.daemon = True  # Daemonize the thread (will terminate with the main program)
        switch_thread.start()

    def stop_switching(self):
        # Stop the switching thread
        self.running = False


class LEDControl:
    def __init__(self, 
                serial_port: str):
        # Open the serial connection
        self.serial = serial.Serial(port=serial_port, 
                                    baudrate=115200)
        time.sleep(2)  # Wait for the Arduino to initialize
    def led_on(self):
        # Send signals to the Arduino
        self.serial.write(b'1')

    def led_off(self):
        # Send signals to the Arduino
        self.serial.write(b'0')

    def close_serial_port(self):
        # Close serial port
        self.serial.close()

if __name__ == "__main__":
    # cam = CameraControl(cam_id=1, 
    #                     exposure_mode="manual")
    # optical_skin_control = OpticalSkinControl(cam, 
    #                                           serial_port='COM5',
    #                                           pdlc_type="normal")
    # while cam.isOpened():
    #     frame = cam.read()
    #     cv2.imshow("RGB Image", frame)
    #     if cv2.waitKey(20) & 0xFF == ord('q'):
    #         break
    
    # cam.release()

    LED_PORT = '/dev/ttyUSB0'
    led_control = LEDControl(serial_port = LED_PORT)
    for _ in range(10):
        led_control.led_on()
        time.sleep(1)
        led_control.led_off()
        time.sleep(1)
