import sys, time, threading, logging, traceback, random, string
from regressor_test import Regressor_test

def regression_test_launcher(ip1, ip2, thread_id):
    try:
        test = Regressor_test(ip1, ip2, thread_id)
        test.start_test()
    except:
        logging.exception("\n**********************EXCEPTION IN THREAD " + thread_id +  " **********************\n")

def random_string(length=6):   
    characters = string.letters
    random_string =  ''.join(random.choice(characters) for i in range(length))
    return random_string

if __name__ == "__main__":
    logging.basicConfig(filename='errors.log',level=logging.DEBUG)
    ips_path = sys.argv[1]
    ip_sets = []
    with open(ips_path, 'r') as ip_file:
        for line in ip_file:
            line = line.strip("\n")
            ip1, ip2 = line.split(" ")
            ip_sets.append((ip1,ip2))

    thread_pool = []
    for index, ip_set in enumerate(ip_sets):
        print "starting thread " + str(index)
        thread_id = random_string()
        thread = threading.Thread(target=regression_test_launcher, args=(ip_set[0],ip_set[1], thread_id))
        thread.start()
        time.sleep(1)
        thread_pool.append(thread)

    for index, thread in enumerate(thread_pool):
        print "waiting for thread " + str(index)
        thread.join()
        print "thread " + str(index) + " finished." 
    