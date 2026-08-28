import can 
import cantools 

# Connect canalyzer tool to simulate vehicle transmission. Then, follow this steps:
# 1) On pi check if CAN hat is working properly
# ip link show can0

# 2) If CAn0 exists, let's setup this with dbc settings (Es. baudrate)
# sudo ip link set can0 down
# sudo ip link set can0 up type can baudrate 500000
# ip -details link show can0
# ---> can0  560   [8]  01 00 00 00 00 00 00 00

DBC_FILE = "BODY_CAN.dbc"

db = cantools.database.load_file(DBC_FILE)

FRAME_IDS = [0x560, 0x526]

bus = can.Bus(
    interface="socketcan",
    channel="can0", #Channel ID of CAN frames
    can_filters=[
        {
            "can_id": FRAME_IDS, #CAN frame ID
            "can_mask": 0x1FFFFFFF, #29-bit for the IDs
            "extended": True #CAN standard or extended
        }
    ]
)

try:
    while True:
        msg = bus.recv()

        if msg is None:
            continue

        decoded = db.decode_message(
            msg.arbitration_id,
            msg.data
        )

        if "KeySts" in decoded:
            print(f"KeySts = {decoded['KeySts']}")

finally:
    bus.shutdown()