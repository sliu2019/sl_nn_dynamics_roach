import command
from struct import pack,unpack
import time,sys,os,traceback

# Path to imageproc-settings repo must be added
sys.path.append(os.path.dirname("../../imageproc-settings/"))
sys.path.append(os.path.dirname("../imageproc-settings/"))      # Some projects have a single-directory structure
import shared_multi as shared

'''FORMAT OF THE IMU DATA
   int32_t posL;
   int32_t posR; 
   int16_t gyroX;
   int16_t gyroY;
   int16_t gyroZ;
   int16_t bemfL;
   int16_t bemfR;
   int16_t Vbatt;'''


#Dictionary of packet formats, for unpack()
pktFormat = { \
    command.TX_DUTY_CYCLE:          'l3f', \
    
    #command.GET_IMU_DATA:           '=l6h', \
    #command.GET_IMU_DATA:	    'ii' + 'hhhhhh', \
    command.GET_IMU_DATA: 	    '=LL' +'4l'+'11h',\
    #command.STREAM_TELEMETRY:       'b', \
    
    command.TX_SAVED_STATE_DATA:    'l3f', \
    command.SET_THRUST_OPEN_LOOP:   '', \
    command.PID_START_MOTORS:       '', \
    command.SET_PID_GAINS:          '10h', \
    command.GET_PID_TELEMETRY:      '', \
    command.GET_AMS_POS:            '=2l', \
    #command.GET_IMU_LOOP_ZGYRO:     '='+2*'Lhhh', \
    command.SET_MOVE_QUEUE:         '', \
    command.SET_STEERING_GAINS:     '6h', \
    #command.SOFTWARE_RESET:         '', \
    command.ERASE_SECTORS:          'L', \
    command.FLASH_READBACK:         '=LL' +'4l'+'11h', \
    command.SLEEP:                  'b', \
    command.ECHO:                   'c' ,\
    command.SET_VEL_PROFILE:        '8h' ,\
    command.WHO_AM_I:               '', \
    command.ZERO_POS:               '=2l', \
    }

               
#XBee callback function, called every time a packet is recieved
def xbee_received(packet):
    rf_data = packet.get('rf_data')
    #rssi = ord(packet.get('rssi'))
    (src_addr, ) = unpack('>H', packet.get('source_addr'))
    #id = packet.get('id')
    #options = ord(packet.get('options'))
    
    #Only print pertinent SRC lines
    #This also allows us to turn off messages on the fly, for telem download
    for r in shared.ROBOTS:
        if r.DEST_ADDR_int == src_addr:
            if r.VERBOSE:
                print "SRC: 0x%04X | " % src_addr,
   
    status = ord(rf_data[0])
    type = ord(rf_data[1])
    data = rf_data[2:]
    
    
    #Record the time the packet is received, so command timeouts
    # can be done
    shared.last_packet_time = time.time()
    try:
        pattern = pktFormat[type]
    except KeyError:
        print "Got bad packet type: ",type
        return
    
    try:
        # GET_IMU_DATA
        if type == command.GET_IMU_DATA:
            datum = unpack(pattern, data)
            shared.callback += 1
            ###print "Packet received from xbee... Got IMU data: ", datum
            shared.imu_queues[src_addr].put(datum)
            #if shared.prevData == datum[0]:
            #    print "idk why, DOUBLE SENT"
            #    print shared.timeBack
            #else:
            #    shared.prevData = datum[0]
            #print shared.callbackIndex, shared.commandIndex
            #    if not shared.callbackIndex in shared.timeBack:
            #        shared.callbackIndex -= 1
            # 	shared.timeBack[shared.callbackIndex] -= time.time()
            #     shared.callbackIndex += 1
            #     if datum[0] % 50 == 0:
            #         print shared.timeBack, shared.callbackIndex, shared.commandIndex
            #     if (datum[0] != -1):
            #         shared.imu_queues[src_addr].put(datum)
        if type == command.STREAM_TELEMETRY:
            datum = unpack(pattern, data)
            if (datum[0] != -1):
                shared.imu_queues[src_addr].put(datum)
                print "Ack StreamTelemetry:",datum
        
        # TX_SAVED_STATE_DATA
        elif type == command.TX_SAVED_STATE_DATA:
            datum = unpack(pattern, data)
            if (datum[0] != -1):
                statedata.append(datum)
         # TX_DUTY_CYCLE
        elif type == command.TX_DUTY_CYCLE:
            datum = unpack(pattern, data)
            if (datum[0] != -1):
                dutycycles.append(datum)
        # ECHO
        elif type == command.ECHO:
            print "echo: status = ",status," type=",type," data = ",data
            
        # SET_PID_GAINS
        elif type == command.SET_PID_GAINS:
            gains = unpack(pattern, data)
            print "Set motor gains to ", gains
            for r in shared.ROBOTS:
                if r.DEST_ADDR_int == src_addr:
                    r.motor_gains_set = True
        
        # FLASH_READBACK
        elif type == command.FLASH_READBACK:
            #shared.pkts = shared.pkts + 1
            #print "Special Telemetry Data Packet, ",shared.pkts
            datum = unpack(pattern, data)
            datum = list(datum)
            #telemTime = datum.pop(0) #pop removes this from data array
            #print "Special Telemetry Data Packet #",telem_index
            print(datum)
            telem_index = datum.pop(1)
            if telem_index > shared.lastTelem and telem_index - shared.lastTelem < 50:
                shared.lastTelem = telem_index

            print(shared.lastTelem)
            #print datum
            if (datum[0] != -1) and (telem_index) >= 0:
                for r in shared.ROBOTS:
                    if r.DEST_ADDR_int == src_addr:
                        if telem_index <= r.numSamples:
                            r.telemtryData[telem_index] = datum
                        else:
                            print "Got out of range telem_index =",telem_index
        
        # ERASE_SECTORS
        elif type == command.ERASE_SECTORS:
            datum = unpack(pattern, data)
            print "Erased flash for", datum[0], " samples."
            if datum[0] != 0:
                for r in shared.ROBOTS:
                    if r.DEST_ADDR_int == src_addr:
                        r.flash_erased = datum[0] 
            
        # SLEEP
        elif type == command.SLEEP:
            datum = unpack(pattern, data)
            print "Sleep reply: ",datum[0]
            if datum[0] == 0:
                shared.awake = True;
        # ZERO_POS
        elif type == command.ZERO_POS:
            print 'AMS zeros established; Previous motor positions:',
            motor = unpack(pattern,data)
            print motor
            print '.... should be 0 now'
            
        # SET_VEL_PROFILE
        elif (type == command.SET_VEL_PROFILE):
            print "Set Velocity Profile readback:"
            temp = unpack(pattern, data)
            print temp
            
        # WHO_AM_I
        elif (type == command.WHO_AM_I):
            print "query : ",data
            for r in shared.ROBOTS:
                if r.DEST_ADDR_int == src_addr:
                    r.robot_queried = True 

    except KeyboardInterrupt:
        print "\nRecieved Ctrl+C in callbackfunc, exiting."
    except Exception as args:
        print "\nGeneral exception from callbackfunc:",args
        print "\n    ******    TRACEBACK    ******    "
        traceback.print_exc()
        print "    *****************************    \n"
        print "Attempting to exit cleanly..."
        shared.xb.halt()
        shared.ser.close()
        sys.exit()
