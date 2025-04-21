class AgentTools:
    def __init__(self):
        pass  # No initialization needed for now

    def calculate(self, what):
        return eval(what)

    def average_dog_weight(self, name):
        name = name.strip().lower()
        if "scottish terrier" in name:
            return "Scottish Terriers average 20 lbs"
        elif "border collie" in name:
            return "A Border Collie's average weight is 37 lbs"
        elif "toy poodle" in name:
            return "A Toy Poodle's average weight is 7 lbs"
        else:
            return "An average dog weighs 50 lbs"

    def get_known_actions(self):
        return {
            "calculate": self.calculate,
            "average_dog_weight": self.average_dog_weight
        }
