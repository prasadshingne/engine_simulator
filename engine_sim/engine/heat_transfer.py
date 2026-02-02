"""Engine heat transfer calculations."""

from dataclasses import dataclass
from typing import Tuple
from .geometry import GeometryParams

@dataclass
class WoschniParams:
    """Parameters for simplified Woschni correlation."""
    C: float = 130.0        # Heat transfer coefficient multiplier
    C_scale: float = 1.5    # Overall scaling factor
    vol_exp: float = 0.6    # Volume exponent
    press_exp: float = 0.8  # Pressure exponent
    temp_exp: float = -0.4  # Temperature exponent
    vel_exp: float = 0.8    # Velocity exponent
    vel_offset: float = 1.4 # Velocity offset term

class HeatTransfer:
    """Heat transfer model using simplified Woschni correlation."""
    
    def __init__(self, geom: GeometryParams, params: WoschniParams = WoschniParams()):
        """
        Initialize heat transfer model.
        
        Parameters
        ----------
        geom : GeometryParams
            Engine geometry parameters
        params : WoschniParams, optional
            Heat transfer correlation parameters
        """
        self.geom = geom
        self.params = params
    
    def heat_transfer_rate(self, crank_angle: float, rpm: float,
                          pressure: float, temperature: float,
                          p_motored: float, p_ref: float,
                          T_ref: float, V_ref: float,
                          T_wall: float) -> Tuple[float, float]:
        """
        Calculate heat transfer rate using simplified Woschni correlation.
        
        Parameters
        ----------
        crank_angle : float
            Crank angle [rad]
        rpm : float
            Engine speed [rev/min]
        pressure : float
            Cylinder pressure [Pa]
        temperature : float
            Gas temperature [K]
        p_motored : float
            Motored pressure at same crank angle [Pa]
        p_ref : float
            Reference pressure at IVC [Pa]
        T_ref : float
            Reference temperature at IVC [K]
        V_ref : float
            Reference volume at IVC [m³]
        T_wall : float
            Wall temperature [K]
            
        Returns
        -------
        Tuple[float, float]
            Heat transfer rate [W], Heat transfer coefficient [W/(m²·K)]
        """
        # Calculate mean piston speed [m/s]
        Up = 2 * self.geom.stroke * rpm / 60
        
        # Calculate volume
        V = self.geom.cylinder_volume(crank_angle)
        
        # Calculate heat transfer coefficient using simplified correlation
        h = (self.params.C_scale * self.params.C * 
             V**self.params.vol_exp * 
             (pressure/1000)**self.params.press_exp * 
             temperature**self.params.temp_exp * 
             (Up + self.params.vel_offset)**self.params.vel_exp)
        
        # Calculate surface areas
        head_area, piston_area, liner_area = self.geom.surface_area(crank_angle)
        total_area = head_area + piston_area + liner_area
        
        # Calculate heat transfer rate
        Q = h * total_area * (temperature - T_wall)
        
        return Q, h 