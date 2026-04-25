from __future__ import annotations

from typing import Any


def _scale_x(x: float, world_min: float, world_max: float, width: int, margin: int) -> float:
    span = max(world_max - world_min, 1.0)
    return margin + ((x - world_min) / span) * (width - 2 * margin)


def build_road_svg(state: dict[str, Any], title: str = "Platoon") -> str:
    width = 980
    height = 260
    margin = 40

    vehicles = state.get("vehicles", {})
    xs = [float(v.get("x", 0.0)) for v in vehicles.values()] if vehicles else [0.0, 100.0]
    world_min = min(xs) - 40.0
    world_max = max(xs) + 40.0

    color_map = {0: "#8f8f8f", 1: "#2f7dff", 2: "#1db954"}

    lane_top = 90
    lane_height = 80

    chunks: list[str] = []
    chunks.append(
        f"<svg width='{width}' height='{height}' xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {width} {height}'>"
    )
    chunks.append("<defs><linearGradient id='bg' x1='0' x2='1' y1='0' y2='1'>")
    chunks.append("<stop offset='0%' stop-color='#f8fafc'/><stop offset='100%' stop-color='#e2e8f0'/></linearGradient></defs>")
    chunks.append("<rect x='0' y='0' width='100%' height='100%' fill='url(#bg)'/>")
    chunks.append(
        f"<rect x='{margin}' y='{lane_top}' width='{width - 2 * margin}' height='{lane_height}' rx='10' fill='#111827' opacity='0.9'/>"
    )
    chunks.append(
        f"<line x1='{margin}' y1='{lane_top + lane_height / 2}' x2='{width - margin}' y2='{lane_top + lane_height / 2}' stroke='#f59e0b' stroke-width='3' stroke-dasharray='18 16'/>"
    )
    chunks.append(f"<text x='{margin}' y='35' font-size='22' font-family='Verdana' fill='#0f172a'>{title}</text>")
    chunks.append(
        f"<text x='{margin}' y='58' font-size='13' font-family='Verdana' fill='#334155'>Step {state.get('timestep', 0)} | Phase: {state.get('phase', 'steady')}</text>"
    )

    if state.get("collision"):
        chunks.append("<rect x='0' y='0' width='100%' height='100%' fill='#dc2626' opacity='0.16'/>")
        chunks.append("<text x='760' y='35' font-size='20' font-family='Verdana' fill='#b91c1c'>COLLISION</text>")

    for car_id in sorted(vehicles.keys()):
        vehicle = vehicles[car_id]
        x_pos = _scale_x(float(vehicle.get("x", 0.0)), world_min, world_max, width, margin)
        car_px_len = 52
        car_px_h = 24
        y = lane_top + (lane_height - car_px_h) / 2
        fill = color_map.get(int(car_id), "#64748b")

        chunks.append(
            f"<rect x='{x_pos - car_px_len}' y='{y}' width='{car_px_len}' height='{car_px_h}' rx='6' fill='{fill}'/>"
        )
        chunks.append(
            f"<text x='{x_pos - car_px_len}' y='{y - 8}' font-size='12' font-family='Verdana' fill='#0f172a'>Car {car_id} | v={float(vehicle.get('velocity', 0.0)):.2f} m/s</text>"
        )

    if 0 in vehicles and 1 in vehicles:
        gap_01 = float(vehicles[0]["x"] - vehicles[1]["x"] - vehicles[1].get("length", 4.5))
        chunks.append(
            f"<text x='{margin}' y='210' font-size='12' font-family='Verdana' fill='#0f172a'>Gap(0->1): {gap_01:.2f} m</text>"
        )

    if 1 in vehicles and 2 in vehicles:
        gap_12 = float(vehicles[1]["x"] - vehicles[2]["x"] - vehicles[2].get("length", 4.5))
        chunks.append(
            f"<text x='{margin + 170}' y='210' font-size='12' font-family='Verdana' fill='#0f172a'>Gap(1->2): {gap_12:.2f} m</text>"
        )

    chunks.append("</svg>")
    return "".join(chunks)
