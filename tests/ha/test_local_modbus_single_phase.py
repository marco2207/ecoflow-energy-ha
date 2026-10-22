"""A Local entry on the single-phase register map.

The single-phase PowerOcean (also sold as the OCEAN 2 Single-Phase Hybrid
Inverter) is read with the registers of EcoFlow's single-phase Modbus table
only. What is covered: the poll asks for no other register, the entry creates
exactly the entities its map publishes, and there is no Modbus control.
"""

from __future__ import annotations

import struct
from typing import Any
from unittest.mock import patch

import pytest
from homeassistant.config_entries import ConfigEntryState
from homeassistant.const import CONF_HOST, CONF_PORT
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import entity_registry as er
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.ecoflow_energy.const import (
    CONF_DEVICES,
    CONF_MODBUS_VARIANT,
    CONF_MODE,
    CONF_UNIT_ID,
    DEVICE_TYPE_POWEROCEAN,
    DOMAIN,
    MODE_LOCAL,
    POWEROCEAN_LOCAL_SENSOR_DEFS,
)
from custom_components.ecoflow_energy.ecoflow.modbus_local import (
    BACKUP_RATIO_OFFSET,
)
from custom_components.ecoflow_energy.ecoflow.parsers.powerocean_modbus import (
    SETUP_BLOCKS,
    SINGLE_PHASE_POLL_BLOCKS,
    VARIANT_SINGLE_PHASE,
    published_keys,
)

from .test_local_modbus_coordinator import StubClient, _encode

SERIAL = "J329TEST00000001"

# Offset -> (wire type, value), from the single-phase protocol table.
_VALUES: dict[int, tuple[str, float]] = {
    0x0206: ("f32", 640.0),  # load
    0x0208: ("f32", 120.0),  # grid, import
    0x020A: ("f32", 2100.0),  # solar
    0x020C: ("f32", 1580.0),  # battery, charging
    0x020E: ("u16", 71),  # SoC
    0x0211: ("u32", 1 << 12),  # status: BMS connected
    0x0213: ("u32", 0),  # system state 2: no alert
    0x0217: ("u16", 15),  # backup ratio
    0x0219: ("u32", 6000),  # max feed power
    0x021C: ("u16", 50),  # indicator brightness
    0x0227: ("u32", 10240),  # battery capacity
    0x022E: ("f32", 2.6),  # grid phase A current
    0x0234: ("f32", 229.8),  # phase A voltage
    0x0800: ("u16", 0),  # fault count
    0x0820: ("u16", 2),  # batteries online
    0x0840: ("f32", -1.5),  # metered energy today
    0x0850: ("f32", 3.25),  # inverter energy today
    0x0860: ("f32", 4.0),  # battery energy today
}


def _single_phase_frame() -> dict[int, bytes]:
    """One full answer: identity blocks plus the single-phase poll blocks."""
    blocks = {start: bytearray(2 * count) for start, count in SINGLE_PHASE_POLL_BLOCKS}
    for offset, (kind, value) in _VALUES.items():
        data = _encode(kind, value)
        for start, count in SINGLE_PHASE_POLL_BLOCKS:
            if start <= offset and offset + len(data) // 2 <= start + count:
                low = (offset - start) * 2
                blocks[start][low : low + len(data)] = data
    answer: dict[int, bytes] = {start: bytes(raw) for start, raw in blocks.items()}
    answer[0x0000] = struct.pack(">HHH", 1, 2, 2)
    answer[0x0003] = SERIAL.encode("ascii")
    # Firmware 6.39.73.12 (0x0627490C), word-swapped on the wire.
    answer[0x000B] = bytes([73, 12, 6, 39])
    return answer


def _single_phase_entry() -> MockConfigEntry:
    return MockConfigEntry(
        domain=DOMAIN,
        version=3,
        data={
            CONF_MODE: MODE_LOCAL,
            CONF_HOST: "modbus.example.test",
            CONF_PORT: 502,
            CONF_UNIT_ID: 1,
            CONF_MODBUS_VARIANT: VARIANT_SINGLE_PHASE,
            CONF_DEVICES: [
                {
                    "sn": SERIAL,
                    "name": "PowerOcean",
                    "product_name": "PowerOcean",
                    "device_type": DEVICE_TYPE_POWEROCEAN,
                    "online": 1,
                }
            ],
        },
    )


async def _setup(hass: HomeAssistant) -> tuple[MockConfigEntry, StubClient, Any]:
    stub = StubClient([_single_phase_frame()])
    entry = _single_phase_entry()
    entry.add_to_hass(hass)
    with patch(
        "custom_components.ecoflow_energy.coordinator.local_modbus.create_link",
        return_value=stub,
    ):
        assert await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()
    assert entry.state is ConfigEntryState.LOADED
    return entry, stub, hass.data[DOMAIN][entry.entry_id][SERIAL]


async def test_the_poll_reads_only_the_single_phase_blocks(
    hass: HomeAssistant,
) -> None:
    entry, stub, coordinator = await _setup(hass)

    assert stub.requested == [SETUP_BLOCKS + SINGLE_PHASE_POLL_BLOCKS]
    data = coordinator.data
    assert data["solar_w"] == pytest.approx(2100.0)
    assert data["soc_pct"] == 71
    assert data["local_grid_phase_a_voltage_v"] == pytest.approx(229.8)
    assert data["local_meter_energy_today_kwh"] == pytest.approx(-1.5)
    assert data["ems_feed_power_limit_w"] == 6000
    assert data["local_bms_connected"] is True
    assert data["local_system_alerts"] == "none"
    assert "modbus_control_active" not in data
    assert coordinator.device_info["sw_version"] == "6.39.73.12"
    await hass.config_entries.async_unload(entry.entry_id)


async def test_the_entry_creates_the_entities_its_map_publishes_and_no_control(
    hass: HomeAssistant,
) -> None:
    entry, _stub, _coordinator = await _setup(hass)

    registry = er.async_get(hass)
    registered = er.async_entries_for_config_entry(registry, entry.entry_id)
    by_domain: dict[str, set[str]] = {}
    for item in registered:
        by_domain.setdefault(item.domain, set()).add(item.unique_id)

    keys = published_keys(VARIANT_SINGLE_PHASE)
    expected_sensors = {
        f"{SERIAL}_{d.key}" for d in POWEROCEAN_LOCAL_SENSOR_DEFS if d.key in keys
    }
    expected_sensors.add(f"{SERIAL}_connection_mode")
    assert by_domain["sensor"] == expected_sensors
    for key in (
        "local_grid_phase_a_voltage_v",
        "local_grid_phase_a_current_a",
        "local_meter_energy_today_kwh",
        "local_inverter_energy_today_kwh",
        "local_battery_energy_today_kwh",
    ):
        assert f"{SERIAL}_{key}" in by_domain["sensor"], key
    for key in ("pcs_ac_freq_hz", "solar_lifetime_energy_kwh", "batt_voltage_v"):
        assert f"{SERIAL}_{key}" not in by_domain["sensor"], key
    assert "switch" not in by_domain
    assert by_domain["binary_sensor"] == {
        f"{SERIAL}_local_off_grid",
        f"{SERIAL}_local_system_abnormal",
        f"{SERIAL}_local_bms_connected",
    }
    assert by_domain["number"] == {
        f"{SERIAL}_local_backup_reserve",
        f"{SERIAL}_local_indicator_brightness",
    }
    await hass.config_entries.async_unload(entry.entry_id)


async def test_modbus_control_is_refused_and_writes_nothing(
    hass: HomeAssistant,
) -> None:
    entry, stub, coordinator = await _setup(hass)

    assert coordinator.control_supported is False
    with pytest.raises(HomeAssistantError):
        await coordinator.async_set_control(True)
    assert stub.writes == []
    assert coordinator.control_enabled is False
    await hass.config_entries.async_unload(entry.entry_id)


async def test_backup_reserve_is_written_and_read_back(hass: HomeAssistant) -> None:
    entry, stub, coordinator = await _setup(hass)

    await coordinator.async_write_register("ems_backup_ratio_pct", 30)

    assert stub.writes == [(BACKUP_RATIO_OFFSET, 30)]
    assert coordinator.data["ems_backup_ratio_pct"] == 30
    await hass.config_entries.async_unload(entry.entry_id)
