"""Entry point to the authoritative production solar-component refinement."""
from circle_arc_detector import SOLAR_COMPONENT_KERNEL, refine_solar_component_mask

__all__ = ["SOLAR_COMPONENT_KERNEL", "refine_solar_component_mask"]
