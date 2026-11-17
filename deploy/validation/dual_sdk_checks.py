"""Pure SDK trace-template tests; never import or initialize real DDS."""

import os
from types import SimpleNamespace
from xml.etree import ElementTree

import pytest
from deploy.backends.sdk import initialize_channel_factory

TEMPLATE = """<?xml version="1.0" encoding="UTF-8" ?>
<CycloneDDS>
  <Domain Id="any">
    <General><Interfaces>
      <NetworkInterface name="$__IF_NAME__$" priority="default" multicast="default"/>
    </Interfaces></General>
    <Tracing><Verbosity>config</Verbosity><OutputFile>/tmp/cdds.LOG</OutputFile></Tracing>
  </Domain>
</CycloneDDS>"""


def test_dual_trace_is_per_actor_and_process_and_preserves_domain_interface(tmp_path):
  channel = SimpleNamespace(ChannelConfigHasInterface=TEMPLATE)
  calls = []

  def initialize(domain, interface):
    calls.append((domain, interface, channel.ChannelConfigHasInterface))

  channel.ChannelFactoryInitialize = initialize
  outputs = []
  for robot_id in ("P1", "P2"):
    outputs.append(
      initialize_channel_factory(
        channel,
        dict(
          robot_id=robot_id,
          domain_id=0,
          network_interface="eno1",
          diagnostic_output_dir=str(tmp_path / "logs & metadata"),
        ),
      )
    )
    assert channel.ChannelConfigHasInterface == TEMPLATE
  assert outputs[0] != outputs[1]
  for robot_id, trace, (domain, interface, xml) in zip(
    ("P1", "P2"), outputs, calls, strict=True
  ):
    assert domain == 0 and interface == "eno1"
    root = ElementTree.fromstring(xml)
    assert root.find("Domain").attrib == {"Id": "any"}
    net = root.find("Domain/General/Interfaces/NetworkInterface")
    assert net.attrib == dict(
      name="$__IF_NAME__$", priority="default", multicast="default"
    )
    assert root.find("Domain/Tracing/Verbosity").text == "config"
    assert root.find("Domain/Tracing/OutputFile").text == trace
    assert trace.endswith(f"cdds-g1_{robot_id.lower()}-{os.getpid()}.LOG")


def test_legacy_sdk_initialization_does_not_read_or_modify_template():
  calls = []
  channel = SimpleNamespace(ChannelFactoryInitialize=lambda *args: calls.append(args))
  assert (
    initialize_channel_factory(channel, dict(domain_id=0, network_interface="eno1"))
    is None
  )
  assert calls == [(0, "eno1")]
  assert not hasattr(channel, "ChannelConfigHasInterface")


def test_sdk_template_restored_even_when_initialization_fails(tmp_path):
  def fail(*_args):
    raise RuntimeError("synthetic initialization failure")

  channel = SimpleNamespace(
    ChannelConfigHasInterface=TEMPLATE, ChannelFactoryInitialize=fail
  )
  with pytest.raises(RuntimeError, match="synthetic"):
    initialize_channel_factory(
      channel,
      dict(
        robot_id="P1",
        domain_id=0,
        network_interface="eno1",
        diagnostic_output_dir=str(tmp_path),
      ),
    )
  assert channel.ChannelConfigHasInterface == TEMPLATE


def test_unknown_sdk_template_fails_before_initialization(tmp_path):
  calls = []
  channel = SimpleNamespace(
    ChannelConfigHasInterface="<CycloneDDS/>",
    ChannelFactoryInitialize=lambda *args: calls.append(args),
  )
  with pytest.raises(ValueError, match="exactly one"):
    initialize_channel_factory(
      channel,
      dict(
        robot_id="P1",
        domain_id=0,
        network_interface="eno1",
        diagnostic_output_dir=str(tmp_path),
      ),
    )
  assert not calls
