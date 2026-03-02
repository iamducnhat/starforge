#!/usr/bin/env python3
"""
World Diff and Command Generation Module for Minecraft
Detects player-built structures by comparing against natural blocks,
and generates Minecraft commands to recreate builds.
"""

import os
import sys

try:
 import anvil
except ImportError:
 print("Error: anvil module required. Install with: pip install anvil-parser")
 sys.exit(1)

# Natural blocks that can generate in world (simplified list)
NATURAL_BLOCKS = {
 'minecraft:stone', 'minecraft:dirt', 'minecraft:grass_block', 'minecraft:bedrock',
 'minecraft:water', 'minecraft:lava', 'minecraft:sand', 'minecraft:gravel',
 'minecraft:coal_ore', 'minecraft:iron_ore', 'minecraft:gold_ore', 'minecraft:diamond_ore',
 'minecraft:redstone_ore', 'minecraft:lapis_ore', 'minecraft:copper_ore',
 'minecraft:granite', 'minecraft:diorite', 'minecraft:andesite', 'minecraft:tuff',
 'minecraft:deepslate', 'minecraft:cobblestone', 'minecraft:mossy_cobblestone',
 'minecraft:oak_log', 'minecraft:spruce_log', 'minecraft:birch_log', 'minecraft:jungle_log',
 'minecraft:acacia_log', 'minecraft:dark_oak_log', 'minecraft:oak_leaves',
 'minecraft:spruce_leaves', 'minecraft:birch_leaves', 'minecraft:jungle_leaves',
 'minecraft:acacia_leaves', 'minecraft:dark_oak_leaves', 'minecraft:air',
 'minecraft:snow', 'minecraft:ice', 'minecraft:packed_ice', 'minecraft:blue_ice',
 'minecraft:clay', 'minecraft:sandstone', 'minecraft:red_sandstone',
 'minecraft:netherrack', 'minecraft:basalt', 'minecraft:blackstone',
 'minecraft:glowstone', 'minecraft:end_stone', 'minecraft:obsidian',
 'minecraft:terracotta', 'minecraft:podzol', 'minecraft:mycelium',
 'minecraft:coarse_dirt', 'minecraft:rooted_dirt', 'minecraft:moss_block',
 'minecraft:calcite', 'minecraft:amethyst_block', 'minecraft:budding_amethyst',
 'minecraft:raw_iron_block', 'minecraft:raw_copper_block', 'minecraft:raw_gold_block',
 'minecraft:large_fern', 'minecraft:tall_grass', 'minecraft:fern', 'minecraft:grass',
 'minecraft:poppy', 'minecraft:dandelion', 'minecraft:rose_bush', 'minecraft:peony',
 'minecraft:lilac', 'minecraft:sunflower', 'minecraft:brown_mushroom', 'minecraft:red_mushroom',
 'minecraft:cactus', 'minecraft:dead_bush', 'minecraft:sugar_cane', 'minecraft:bamboo',
 'minecraft:vine', 'minecraft:lily_pad', 'minecraft:sea_grass', 'minecraft:kelp',
 'minecraft:pointed_dripstone', 'minecraft:sculk', 'minecraft:sculk_sensor',
 'minecraft:sculk_shrieker', 'minecraft:sculk_catalyst'
}


def get_world_diff(region_file, cx, cz, y_min=-64, y_max=320):
 """
 Compares current chunk blocks against natural terrain to find player modifications.
 
 Args:
 region_file: Path to .mca region file
 cx: Chunk X coordinate (0-31 within region)
 cz: Chunk Z coordinate (0-31 within region)
 y_min: Minimum Y to scan (default -64 for 1.18+)
 y_max: Maximum Y to scan (default 320 for 1.18+)
 
 Returns:
 Dict with 'player_blocks' list and 'stats' dict
 """
 if not os.path.exists(region_file):
 return {"error": f"File {region_file} not found."}

 try:
 region = anvil.Region.from_file(region_file)
 chunk = anvil.Chunk.from_region(region, cx, cz)
 
 player_blocks = []
 natural_blocks = []
 
 for section in chunk.sections:
 if section is None:
 continue
 section_y = section.y * 16
 
 for y in range(16):
 global_y = section_y + y
 if global_y < y_min or global_y > y_max:
 continue
 for z in range(16):
 for x in range(16):
 try:
 block = section.get_block(x, y, z)
 block_name = f"{block.namespace}:{block.id}"
 global_x = (cx * 16) + x
 global_z = (cz * 16) + z
 if block.id == 'air':
 continue
 if block_name not in NATURAL_BLOCKS:
 player_blocks.append({
 'x': global_x, 'y': global_y, 'z': global_z,
 'block': block_name,
 'properties': block.properties if hasattr(block, 'properties') else {}
 })
 else:
 natural_blocks.append(block_name)
 except Exception:
 continue
 
 return {
 "player_blocks": player_blocks,
 "stats": {
 "total_player_blocks": len(player_blocks),
 "unique_player_blocks": len(set(b['block'] for b in player_blocks)),
 "natural_blocks_scanned": len(natural_blocks)
 }
 }
 except Exception as e:
 return {"error": f"Error computing diff: {e}"}


def generate_build_commands(blocks, relative_coords=False, base_x=0, base_y=0, base_z=0):
 """
 Generates Minecraft setblock commands to place blocks.
 
 Args:
 blocks: List of block dicts from get_world_diff
 relative_coords: If True, use relative coordinates (~)
 base_x/y/z: Base position for relative coordinates
 
 Returns:
 String of setblock commands, one per line
 """
 commands = []
 for b in blocks:
 x, y, z = b['x'], b['y'], b['z']
 block = b['block']
 props = b.get('properties', {})
 
 if relative_coords:
 x = f"~{x - base_x}" if x != base_x else "~"
 y = f"~{y - base_y}" if y != base_y else "~"
 z = f"~{z - base_z}" if z != base_z else "~"
 
 if props:
 prop_str = ','.join(f"{k}={v}" for k, v in props.items())
 block_str = f"{block}[{prop_str}]"
 else:
 block_str = block
 
 commands.append(f"setblock {x} {y} {z} {block_str}")
 return '\n'.join(commands)


def export_build_mcfunction(blocks, filename="player_build.mcfunction", description="Player build"):
 """
 Creates a complete .mcfunction file for a datapack.
 
 Args:
 blocks: List of block dicts from get_world_diff
 filename: Output filename
 description: Description comment in file
 
 Returns:
 Status message string
 """
 content = [f"# {description}", f"# Generated by world_diff.py", f"# Total blocks: {len(blocks)}", ""]
 
 if blocks:
 min_x = min(b['x'] for b in blocks)
 min_y = min(b['y'] for b in blocks)
 min_z = min(b['z'] for b in blocks)
 content.append(f"# Origin: ({min_x}, {min_y}, {min_z})")
 content.append("")
 
 for b in blocks:
 x, y, z = b['x'] - min_x, b['y'] - min_y, b['z'] - min_z
 block = b['block']
 props = b.get('properties', {})
 if props:
 prop_str = ','.join(f"{k}={v}" for k, v in props.items())
 block_str = f"{block}[{prop_str}]"
 else:
 block_str = block
 content.append(f"setblock ~{x} ~{y} ~{z} {block_str}")
 
 with open(filename, 'w') as f:
 f.write('\n'.join(content))
 return f"Exported {len(blocks)} blocks to {filename}"


def scan_region_for_builds(region_file, natural_threshold=50):
 """
 Scans an entire region file for chunks with player modifications.
 
 Args:
 region_file: Path to .mca region file
 natural_threshold: Minimum player blocks to consider a "build"
 
 Returns:
 Dict with region info and list of builds found
 """
 if not os.path.exists(region_file):
 return {"error": f"File {region_file} not found."}

 try:
 region = anvil.Region.from_file(region_file)
 builds_found = []
 
 for cz in range(32):
 for cx in range(32):
 try:
 offset, _ = region.chunk_location(cx, cz)
 if offset == 0:
 continue
 except:
 continue
 
 diff = get_world_diff(region_file, cx, cz)
 if 'error' in diff:
 continue
 if diff['stats']['total_player_blocks'] >= natural_threshold:
 builds_found.append({
 'chunk_x': cx, 'chunk_z': cz,
 'player_blocks': diff['player_blocks'],
 'stats': diff['stats']
 })
 
 return {"region_file": region_file, "builds_found": len(builds_found), "builds": builds_found}
 except Exception as e:
 return {"error": f"Error scanning region: {e}"}


def interactive_mode():
 """Interactive CLI for world diff operations."""
 print("=== Minecraft World Diff Tool ===")
 print("Commands: diff, scan, export, commands, quit")
 
 while True:
 try:
 cmd = input("\n> ").strip().lower()
 except EOFError:
 break
 
 if cmd == 'quit' or cmd == 'exit':
 break
 
 elif cmd == 'diff':
 region_file = input("Region file path: ").strip()
 cx = int(input("Chunk X (0-31): "))
 cz = int(input("Chunk Z (0-31): "))
 result = get_world_diff(region_file, cx, cz)
 if 'error' in result:
 print(f"Error: {result['error']}")
 else:
 print(f"Player blocks found: {result['stats']['total_player_blocks']}")
 print(f"Unique block types: {result['stats']['unique_player_blocks']}")
 if result['stats']['total_player_blocks'] > 0:
 block_types = set(b['block'] for b in result['player_blocks'])
 print(f"Block types: {', '.join(sorted(block_types))}")
 
 elif cmd == 'scan':
 region_file = input("Region file path: ").strip()
 threshold = int(input("Minimum player blocks threshold (default 50): ") or "50")
 result = scan_region_for_builds(region_file, threshold)
 if 'error' in result:
 print(f"Error: {result['error']}")
 else:
 print(f"Builds found: {result['builds_found']}")
 for build in result['builds']:
 print(f" Chunk ({build['chunk_x']}, {build['chunk_z']}): {build['stats']['total_player_blocks']} blocks")
 
 elif cmd == 'export':
 region_file = input("Region file path: ").strip()
 cx = int(input("Chunk X (0-31): "))
 cz = int(input("Chunk Z (0-31): "))
 filename = input("Output filename (default: build.mcfunction): ").strip() or "build.mcfunction"
 desc = input("Description: ").strip() or "Exported build"
 
 result = get_world_diff(region_file, cx, cz)
 if 'error' in result:
 print(f"Error: {result['error']}")
 else:
 msg = export_build_mcfunction(result['player_blocks'], filename, desc)
 print(msg)
 
 elif cmd == 'commands':
 region_file = input("Region file path: ").strip()
 cx = int(input("Chunk X (0-31): "))
 cz = int(input("Chunk Z (0-31): "))
 
 result = get_world_diff(region_file, cx, cz)
 if 'error' in result:
 print(f"Error: {result['error']}")
 else:
 print(f"# {result['stats']['total_player_blocks']} setblock commands:")
 print(generate_build_commands(result['player_blocks']))
 
 else:
 print(f"Unknown command: {cmd}")
 print("Commands: diff, scan, export, commands, quit")


if __name__ == "__main__":
 interactive_mode()
