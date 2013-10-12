class Transit:
    def __init__(self, file_):
        self.name = file_.readline().strip()
        if self.name == '':
            raise IOError
        self.begin = file_.readline().strip()
        self.center = file_.readline().strip()
        self.center2 = file_.readline().strip()
        self.end = file_.readline().strip()
        self.ra = file_.readline().strip()
        self.dec = file_.readline().strip()


def process_file(name='transit-predictions.txt'):
    f = open(name, encoding='utf-8')
    transits = []

    EOF = False
    while not EOF:
        try:
            transits.append(Transit(f))
            print(transits[-1].begin)
        except IOError:
            EOF = True
    print(len(transits))


if __name__ == '__main__':
    transits = process_file()
