import rospy
import time

def main(): 

    print('start Go2 navigation environment node')
    rospy.init_node("navigation_env", anonymous=True)
    try:
        while not rospy.is_shutdown():
            time.sleep(0.1)
    except KeyboardInterrupt:
        print('absc1')
        print('absc2')
        print('absc3')
    finally:
        rospy.signal_shutdown("Condition met, shutting down.")

if __name__ == "__main__":
    main()