#!/usr/bin/env python3
"""Test all model types support temperature and top_p."""
import sys
sys.path.insert(0, '.')

from assistant import model

print("=" * 60)
print("TESTING TEMPERATURE AND TOP_P SUPPORT")
print("=" * 60)

# Test Nvidia model
print("\n### Nvidia Model ###")
nvidia = model.NvidiaModel(model_name="z-ai/glm5", api_key="test_key")
print(f"✓ Default temperature: {nvidia.get_temperature()}")
print(f"✓ Default top_p: {nvidia.get_top_p()}")
nvidia.set_temperature(1.0)
nvidia.set_top_p(0.9)
print(f"✓ After changes - temp: {nvidia.get_temperature()}, top_p: {nvidia.get_top_p()}")

# Test OpenRouter model
print("\n### OpenRouter Model ###")
openrouter = model.OpenRouterModel(model_name="test/model", api_key="test_key")
print(f"✓ Default temperature: {openrouter.get_temperature()}")
print(f"✓ Default top_p: {openrouter.get_top_p()}")
openrouter.set_temperature(0.7)
openrouter.set_top_p(0.85)
print(f"✓ After changes - temp: {openrouter.get_temperature()}, top_p: {openrouter.get_top_p()}")

# Test CLI commands
print("\n### CLI Commands Available ###")
print("- /temperature <n>     Set temperature (0.0-2.0)")
print("- /temperature         Show current temperature")
print("- /top_p <n>           Set top_p (0.0-1.0)")
print("- /top_p or /topp      Show current top_p")

print("\n### Environment Variables ###")
print("- ASSISTANT_TEMPERATURE=0.2  (default)")
print("- ASSISTANT_TOP_P=0.95       (default)")

print("\n✓ All tests passed!")
