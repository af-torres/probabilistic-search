import math



def example1(l: float) -> float:
    return math.sin(5 * l) + math.tan(l)


examples = {
    "example1": {
        "function": example1,
        "name": "L(x) = sin(5x) + tan(x)"
    }
}
