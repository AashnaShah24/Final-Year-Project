def time():
    from datetime import datetime

    now = datetime.now() # current date and time
    time = now.strftime("%H:%M:%S")
    print("time:", time)

    year = now.strftime("%Y")
    print("year:", year)

    month = now.strftime("%m")
    print("month:", month)

    day = now.strftime("%d")
    print("day:", day)

time()