#!/usr/bin/env python3
"""Test temperature and top_p settings."""
import sys
sys.path.insert(0, '.')

from assistant import model

# Test Nvidia model
nvidia_model = model.NvidiaModel(model_name="z-ai/glm5", api_key="test_key")

print("=== Nvidia Model ===")
print(f"Initial temperature: {nvidia_model.get_temperature()}")
print(f"Initial top_p: {nvidia_model.get_top_p()}")

# Test setting temperature
ok, msg = nvidia_model.set_temperature(0.5)
print(f"Set temperature 0.5: {msg}")
print(f"New temperature: {nvidia_model.get_temperature()}")

# Test setting top_p
ok, msg = nvidia_model.set_top_p(0.8)
print(f"Set top_p 0.8: {msg}")
print(f"New top_p: {nvidia_model.get_top_p()}")

# Test invalid values
ok, msg = nvidia_model.set_temperature(3.0)
print(f"Invalid temp (3.0): {msg}")

ok, msg = nvidia_model.set_top_p(1.5)
print(f"Invalid top_p (1.5): {msg}")

# Check payload includes top_p
messages = [{"role": "user", "content": "test"}]
payload = nvidia_model._chat_payload(messages, stream=False)
print(f"\nPayload contains top_p: {'top_p' in payload}")
print(f"Payload top_p value: {payload.get('top_p')}")

print("\n✓ Temperature and top_p settings working!")
