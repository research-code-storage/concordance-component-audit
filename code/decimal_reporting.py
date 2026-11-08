"""Decimal half-up formatting for inference tables and percentages."""
from decimal import Decimal, ROUND_HALF_UP

def rounded(value, places=3, scale=1):
    value=Decimal(str(value))*Decimal(str(scale))
    quantum=Decimal(1).scaleb(-places)
    return format(value.quantize(quantum,rounding=ROUND_HALF_UP),f'.{places}f')

def percent(value, places=1):
    return rounded(value,places,100)+r'\%'
