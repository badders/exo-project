from datetime import datetime, timedelta

class Transit:
    def __init__(self, file_):
        self.name = file_.readline().strip()
        if self.name == '':
            raise IOError

        l2 = file_.readline().strip()
        start_hour, start_minute = l2.split(':')

        l3 = file_.readline().strip()
        start_alt, start_date, center_time = l3.split()
        
        start_day, start_month, _ = start_date.split('.')
        center_hour, center_minute = center_time.split(':')

        l4= file_.readline().strip()
        center_alt, end_time = l4.split()
        end_hour, end_minute = end_time.split(':')

        l5 = file_.readline().strip()
        end_alt, D, mag, t_mag, _ = l5.split()

        def get_angle(data):
            try:
                angle = int(data[:2])
            except ValueError:
                angle = int(data[:1])
            return angle

        self.start_alt = get_angle(start_alt)
        self.center_alt = get_angle(center_alt)
        self.end_alt = get_angle(end_alt)

        self.magnitude = float(mag)
        self.magnitude_dip = float(t_mag)

        start_month = int(start_month)
        year = 2013
        if start_month < 6:
            year = 2014

        self.start_date = datetime(year, start_month, int(start_day), int(start_hour), int(start_minute))

        self.end_date = self.start_date.replace(hour=int(end_hour), minute=int(end_minute))

        if int(end_hour) < int(start_hour):
            self.end_date = self.end_date + timedelta(days=1)

        assert(self.start_date < self.end_date)
        self.ra = file_.readline().strip()
        self.dec = file_.readline().strip()

    def __repr__(self):
        return'{} to {:02}:{:02}\tMag: {:.1f}\tDip: {:0.3f}\tAlt: {}'.format(self.start_date, self.end_date.hour, self.end_date.minute, self.magnitude, self.magnitude_dip, self.start_alt)

def process_file(name='transit-predictions.txt'):
    f = open(name, encoding='utf-8')
    transits = []

    EOF = False
    while not EOF:
        try:
            transits.append(Transit(f))
        except IOError:
            EOF = True
    return transits


def filter_func(transit):
    if transit.magnitude_dip < 0.01:
        return False
    if transit.magnitude > 12:
        return False
    if transit.start_alt > 45:
        return False
    if transit.end_alt > 45:
        return False
    if transit.start_date <= datetime.utcnow():
        return False

    return True

if __name__ == '__main__':
    transits = process_file()
    ts = [t for t in transits if filter_func(t)]
    for t in ts:
        print(t)
