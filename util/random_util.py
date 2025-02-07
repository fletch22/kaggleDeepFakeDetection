import string
import random


def random_string_digits(string_length:int=6) -> str:
  letters_and_digits = string.ascii_letters + string.digits
  return ''.join(random.choice(letters_and_digits) for i in range(string_length))
