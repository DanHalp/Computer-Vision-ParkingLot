"""
this is the main file of the project. and the one that you should run.

"""


from border import Border
from Section import Section
import cv2
from threading import Thread
from time import sleep
from Atomic_bool import AtomicBoolean
import Constants

FLAG = AtomicBoolean()
sections = []

def print_sections_availability():
    global FLAG
    while FLAG.get():
        sleep(5)
        print("Current DATA:")
        for section in sections:
            print("##################")
            print("Section: ", section.name)
            print("Number of cars in section: " , section.total_cars_in_section.value)
            print("Number of available parking spaces: ", section.get_total_available_parking_spaces())

        print()


def main():
    global FLAG
    threads = []
    first_floor_section = Section("first floor", 50)
    second_floor_section = Section("second floor", 30)
    out_section = Section("outside", 100000)
    sections.append(first_floor_section)
    sections.append(second_floor_section)


    net = cv2.dnn.readNetFromCaffe(Constants.PROTOTXT__PATH, Constants.CAFFE_MODEL_PATH)
    net2 = cv2.dnn.readNetFromCaffe(Constants.PROTOTXT__PATH, Constants.CAFFE_MODEL_PATH)


    border_out_1 = Border(first_floor_section, out_section, Constants.VID_PATH+Constants.VID_1_NAME, net, 1)
    border_1_2 = Border(second_floor_section, first_floor_section, Constants.VID_PATH+Constants.VID_2_NAME, net2, 2)
    try:
        thread_border_out_1 = Thread(target=border_out_1.start_counting)
        thread_border_1_2 = Thread(target=border_1_2.start_counting)
        section_thread = Thread(target=print_sections_availability)
        threads.append(thread_border_out_1)
        threads.append(thread_border_1_2)
        # threads.append(section_thread)
        for thread in threads:
            thread.start()
        section_thread.start()
        for thread in threads:
            thread.join()
            print("################finished_join#############")
        FLAG.false()
    except Exception as e:
        print("Unable to create\start threads." + str(e.with_traceback(None)))
        exit(1)



if __name__ == "__main__":
    main()





