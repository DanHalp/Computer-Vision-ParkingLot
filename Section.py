from atomic_counter import AtomicCounter

class Section:
    name = None
    total_parking_spaces_in_section = 0
    total_cars_in_section = None

    def __init__(self, name, num_parking_spaces):
        self.name = name
        self.total_parking_spaces_in_section = num_parking_spaces
        self.total_cars_in_section = AtomicCounter(0)


    def get_total_available_parking_spaces(self):
        return self.total_parking_spaces_in_section - self.total_cars_in_section.value


    def update_car_entered(self):
        self.total_cars_in_section.increment(1)


    def update_car_exited(self):
        self.total_cars_in_section.decrement(1)

