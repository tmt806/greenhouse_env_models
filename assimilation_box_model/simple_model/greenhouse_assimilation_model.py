from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict


@dataclass(frozen=True)
class PhysicalConstants:
    R: float = 8.314462618  # J mol-1 K-1
    R_da: float = 287.05  # J kg-1 K-1
    cp_da: float = 1005.0  # J kg_da-1 K-1
    cp_v: float = 1860.0  # J kg_v-1 K-1
    Lv: float = 2.45e6  # J kg-1


@dataclass(frozen=True)
class GreenhouseParams:
    V: float  # m3
    A_cov: float  # m2
    UA: float  # W K-1
    h_c: float  # W m-2 K-1
    delta_T_sky: float = 0.0  # K
    p: float = 101325.0  # Pa
    delta_for_s: float = 0.1  # degC
    clip_condensation_to_positive: bool = False
    screen_solar_transmittance_closed: float = 0.35  # -
    screen_ua_closed_ratio: float = 0.70  # -
    screen_sky_coupling_closed_ratio: float = 0.30  # -


@dataclass(frozen=True)
class StateInputs:
    T_in: float  # degC
    RH_in: float  # 0-1
    X_in: float  # mol mol-1
    T_out: float  # degC
    RH_out: float  # 0-1
    X_out: float  # mol mol-1
    dT_in_dt: float  # K s-1
    d_omega_in_dt: float  # kg kg-1 s-1
    dX_in_dt: float  # s-1
    H_air: float  # W, burner/heater/etc. except solar shortwave
    S_CO2: float  # mol s-1
    H_solar: float = 0.0  # W, incident solar heat before screen attenuation
    W_inj: float = 0.0  # kg s-1
    screen_closure: float = 0.0  # 0-1


def sat_vapor_pressure_pa(T_c: float) -> float:
    return 611.2 * math.exp(17.67 * T_c / (T_c + 243.5))


def mixing_ratio_from_rh(T_c: float, rh: float, p: float) -> float:
    e = rh * sat_vapor_pressure_pa(T_c)
    return 0.62198 * e / (p - e)


def sat_mixing_ratio(T_c: float, p: float) -> float:
    e_sat = sat_vapor_pressure_pa(T_c)
    return 0.62198 * e_sat / (p - e_sat)


def moist_air_enthalpy(T_c: float, omega: float, cp_da: float, cp_v: float) -> float:
    return cp_da * T_c + omega * (2.501e6 + cp_v * T_c)


def dry_air_density(T_c: float, rh: float, p: float, R_da: float) -> float:
    e = rh * sat_vapor_pressure_pa(T_c)
    return (p - e) / (R_da * (T_c + 273.15))


def molar_density(T_c: float, p: float, R: float) -> float:
    return p / (R * (T_c + 273.15))


def d_omega_sat_dT(T_c: float, p: float, delta: float) -> float:
    om_plus = sat_mixing_ratio(T_c + delta, p)
    om_minus = sat_mixing_ratio(T_c - delta, p)
    return (om_plus - om_minus) / (2.0 * delta)


def clamp01(x: float) -> float:
    return min(max(float(x), 0.0), 1.0)


def screened_transport_params(
    params: GreenhouseParams,
    screen_closure: float,
) -> Dict[str, float]:
    closure = clamp01(screen_closure)
    solar_trans = 1.0 - closure * (1.0 - params.screen_solar_transmittance_closed)
    ua_ratio = 1.0 - closure * (1.0 - params.screen_ua_closed_ratio)
    sky_ratio = 1.0 - closure * (1.0 - params.screen_sky_coupling_closed_ratio)
    return {
        "screen_closure": closure,
        "solar_transmittance": solar_trans,
        "UA_eff": params.UA * ua_ratio,
        "delta_T_sky_eff": params.delta_T_sky * sky_ratio,
    }


def condensation_flux_and_rate(
    T_in: float,
    RH_in: float,
    T_out: float,
    params: GreenhouseParams,
    const: PhysicalConstants,
    screen_closure: float = 0.0,
) -> Dict[str, float]:
    omega_in = mixing_ratio_from_rh(T_in, RH_in, params.p)
    omega_sat_in = sat_mixing_ratio(T_in, params.p)
    delta_omega_sat = omega_in - omega_sat_in

    screened = screened_transport_params(params, screen_closure)
    U = screened["UA_eff"] / params.A_cov
    T_out_eff = T_out - screened["delta_T_sky_eff"]
    delta_T_eff = T_in - T_out_eff
    s = d_omega_sat_dT(T_in, params.p, params.delta_for_s)
    gamma = const.Lv / const.cp_da

    numerator = delta_omega_sat + s * (U / params.h_c) * delta_T_eff
    j = (params.h_c / const.cp_da) * (numerator / (1.0 + gamma * s))
    if params.clip_condensation_to_positive:
        j = max(0.0, j)
    W_cond = params.A_cov * j

    return {
        "j_cond": j,
        "W_cond": W_cond,
        "s": s,
        "omega_sat_in": omega_sat_in,
        "T_out_eff": T_out_eff,
        "UA_eff": screened["UA_eff"],
        "delta_T_sky_eff": screened["delta_T_sky_eff"],
    }


def solve_q_e_p(
    params: GreenhouseParams,
    state: StateInputs,
    const: PhysicalConstants = PhysicalConstants(),
) -> Dict[str, float]:
    omega_in = mixing_ratio_from_rh(state.T_in, state.RH_in, params.p)
    omega_out = mixing_ratio_from_rh(state.T_out, state.RH_out, params.p)
    h_in = moist_air_enthalpy(state.T_in, omega_in, const.cp_da, const.cp_v)
    h_out = moist_air_enthalpy(state.T_out, omega_out, const.cp_da, const.cp_v)
    rho_da = dry_air_density(state.T_in, state.RH_in, params.p, const.R_da)
    rho_mol = molar_density(state.T_in, params.p, const.R)
    screened = screened_transport_params(params, state.screen_closure)
    H_solar_eff = state.H_solar * screened["solar_transmittance"]
    H_in = state.H_air + H_solar_eff

    cond = condensation_flux_and_rate(
        T_in=state.T_in,
        RH_in=state.RH_in,
        T_out=state.T_out,
        params=params,
        const=const,
        screen_closure=state.screen_closure,
    )
    W_cond = cond["W_cond"]

    delta_omega = omega_in - omega_out
    delta_h = h_in - h_out
    L_star = 2.501e6 + const.cp_v * state.T_in
    T_out_eff = cond["T_out_eff"]

    R1 = rho_da * params.V * state.d_omega_in_dt - state.W_inj + W_cond
    R2 = (
        H_in
        - cond["UA_eff"] * (state.T_in - T_out_eff)
        - rho_da * params.V * (const.cp_da + const.cp_v * omega_in) * state.dT_in_dt
        - L_star * (state.W_inj - W_cond)
    )

    denom_q = rho_da * delta_h
    if abs(denom_q) < 1e-9:
        raise ZeroDivisionError(
            "Q calculation is unstable because rho_da * (h_in - h_out) is too small."
        )

    Q = (R2 - L_star * R1) / denom_q
    E = R1 + rho_da * delta_omega * Q
    P = rho_mol * Q * (state.X_out - state.X_in) + state.S_CO2 - rho_mol * params.V * state.dX_in_dt

    return {
        "Q_m3_s": Q,
        "E_kg_s": E,
        "P_mol_s": P,
        "W_cond_kg_s": W_cond,
        "j_cond_kg_m2_s": cond["j_cond"],
        "omega_in": omega_in,
        "omega_out": omega_out,
        "h_in_J_kgda": h_in,
        "h_out_J_kgda": h_out,
        "rho_da_kg_m3": rho_da,
        "rho_mol_mol_m3": rho_mol,
        "delta_h_J_kgda": delta_h,
        "delta_omega_kg_kgda": delta_omega,
        "R1_kg_s": R1,
        "R2_W": R2,
        "L_star_J_kg": L_star,
        "screen_closure": screened["screen_closure"],
        "screen_solar_transmittance": screened["solar_transmittance"],
        "UA_eff_W_K": cond["UA_eff"],
        "delta_T_sky_eff_K": cond["delta_T_sky_eff"],
        "H_air_W": state.H_air,
        "H_solar_raw_W": state.H_solar,
        "H_solar_eff_W": H_solar_eff,
        "H_in_W": H_in,
    }


if __name__ == "__main__":
    params = GreenhouseParams(
        V=3500.0,
        A_cov=2200.0,
        UA=1800.0,
        h_c=5.0,
        delta_T_sky=2.0,
        p=101325.0,
        clip_condensation_to_positive=False,
    )
    state = StateInputs(
        T_in=22.0,
        RH_in=0.78,
        X_in=780e-6,
        T_out=10.0,
        RH_out=0.60,
        X_out=420e-6,
        dT_in_dt=0.0,
        d_omega_in_dt=0.0,
        dX_in_dt=0.0,
        H_air=18000.0,
        H_solar=12000.0,
        S_CO2=0.0015,
        W_inj=0.0,
        screen_closure=0.6,
    )
    out = solve_q_e_p(params, state)
    for k, v in out.items():
        print(f"{k}: {v}")
