import anvil
import nbtlib
import os
import sys
import json
import random
import hashlib

def get_block_at(region_file, x, y, z):
    """
    Returns the block at the given global coordinates.
    x, y, z are global coordinates.
    """
    # Region coordinates
    rx = x // 512
    rz = z // 512
    
    # Chunk coordinates relative to region
    cx = (x % 512) // 16
    cz = (z % 512) // 16
    
    # Block coordinates relative to chunk
    bx = x % 16
    bz = z % 16
    
    if not os.path.exists(region_file):
        return f"Error: File {region_file} not found."

    try:
        region = anvil.Region.from_file(region_file)
        chunk = anvil.Chunk.from_region(region, cx, cz)
        block = chunk.get_block(bx, y, bz)
        return block
    except Exception as e:
        return f"Error reading block: {e}"

def list_blocks_in_chunk(region_file, cx, cz):
    """
    Lists all unique blocks in a specific chunk using the section palettes.
    This is much faster than iterating every block.
    """
    if not os.path.exists(region_file):
        return f"Error: File {region_file} not found."

    try:
        region = anvil.Region.from_file(region_file)
        chunk = anvil.Chunk.from_region(region, cx, cz)
        
        blocks = set()
        # anvil-parser2 chunks have a 'sections' attribute
        # Usually sections are from -4 to 19 (for -64 to 320 height)
        for section in chunk.sections:
            if section and hasattr(section, 'palette'):
                for block in section.palette:
                    if block.id != 'air':
                        blocks.add(f"{block.namespace}:{block.id}")
        return sorted(list(blocks))
    except Exception as e:
        return f"Error reading chunk sections: {e}"

def list_entities_in_chunk(region_file, cx, cz):
    """
    Lists block entities (TileEntities) in a specific chunk.
    Note: This reads block entities like chests/signs, not moving entities (mobs).
    """
    if not os.path.exists(region_file):
        return f"Error: File {region_file} not found."

    try:
        region = anvil.Region.from_file(region_file)
        chunk = anvil.Chunk.from_region(region, cx, cz)
        
        entities = []
        # In anvil-parser2, chunk.tile_entities or chunk.data['TileEntities']
        # We'll check for 'tile_entities' attribute first
        te_list = getattr(chunk, 'tile_entities', None)
        if te_list is None and 'TileEntities' in chunk.data:
            te_list = chunk.data['TileEntities']
            
        if te_list:
            for te in te_list:
                id_val = te.get('id', 'unknown')
                x, y, z = te.get('x'), te.get('y'), te.get('z')
                entities.append(f"{id_val} at ({x}, {y}, {z})")
        
        return entities
    except Exception as e:
        return f"Error reading entities: {e}"

def find_blocks_in_chunk(region_file, cx, cz, block_id):
    """
    Finds all coordinates of a specific block ID within a chunk.
    """
    if not os.path.exists(region_file):
        return f"Error: File {region_file} not found."

    try:
        region = anvil.Region.from_file(region_file)
        chunk = anvil.Chunk.from_region(region, cx, cz)
        
        found = []
        # Iterate through sections (16x16x16 blocks each)
        for section in chunk.sections:
            if section is None: continue
            
            # Check if block exists in this section's palette to skip empty ones
            palette = [b.name() for b in section.palette]
            if not any(block_id in name for name in palette):
                continue

            # Global Y for this section
            y_base = section.y * 16
            for y in range(16):
                for z in range(16):
                    for x in range(16):
                        block = section.get_block(x, y, z)
                        if block_id in block.name():
                            # Convert to local chunk coords or global?
                            # Let's return chunk-relative coords (0-15, y, 0-15)
                            found.append((x, y_base + y, z))
        return found
    except Exception as e:
        return f"Error searching blocks: {e}"

def get_biome_at(region_file, x, y, z):
    """
    Gets the biome at specific global coordinates.
    """
    if not os.path.exists(region_file):
        return f"Error: File {region_file} not found."

    try:
        region = anvil.Region.from_file(region_file)
        # Global to chunk coords
        cx = x // 16
        cz = z // 16
        # Local chunk coords
        lx = x % 16
        lz = z % 16
        
        chunk = anvil.Chunk.from_region(region, cx, cz)
        biome = chunk.get_biome(lx, y, lz)
        return biome
    except Exception as e:
        return f"Error reading biome: {e}"

def get_world_info(level_dat_path):
    """
    Reads global world information from level.dat.
    """
    if not os.path.exists(level_dat_path):
        return f"Error: File {level_dat_path} not found."

    try:
        nbt = nbtlib.load(level_dat_path)
        data = nbt.get('Data', {})
        
        info = {
            "LevelName": str(data.get('LevelName', 'Unknown')),
            "DataVersion": int(data.get('DataVersion', 0)),
            "LastPlayed": int(data.get('LastPlayed', 0)),
        }
        
        # Seed location changed in newer versions
        if 'WorldGenSettings' in data:
            info["Seed"] = int(data['WorldGenSettings'].get('seed', 0))
        else:
            info["Seed"] = int(data.get('RandomSeed', 0))
            
        return info
    except Exception as e:
        return f"Error reading level.dat: {e}"

def list_chunks_in_region(region_file):
    """
    Lists all generated chunks in a region file.
    """
    if not os.path.exists(region_file):
        return f"Error: File {region_file} not found."

    try:
        region = anvil.Region.from_file(region_file)
        present_chunks = []
        for cz in range(32):
            for cx in range(32):
                # anvil-parser2: Region.chunk_location(cx, cz) returns (offset, sector_count)
                # If offset is 0, the chunk is not generated.
                offset, _ = region.chunk_location(cx, cz)
                if offset > 0:
                    present_chunks.append((cx, cz))
        return present_chunks
    except Exception as e:
        return f"Error listing chunks: {e}"

def get_player_data(player_dat_path):
    """
    Reads player information from a <uuid>.dat file.
    """
    if not os.path.exists(player_dat_path):
        return f"Error: File {player_dat_path} not found."

    try:
        nbt = nbtlib.load(player_dat_path)
        # Player data is usually at the root of the NBT file
        pos = [float(x) for x in nbt.get('Pos', [])]
        dim = str(nbt.get('Dimension', 'unknown'))
        xp = int(nbt.get('XpLevel', 0))
        
        inventory = []
        for item in nbt.get('Inventory', []):
            slot = int(item.get('Slot', -1))
            id_val = str(item.get('id', 'unknown'))
            count = int(item.get('Count', 0))
            inventory.append(f"Slot {slot}: {id_val} x{count}")
            
        return {
            "Position": pos,
            "Dimension": dim,
            "XP Level": xp,
            "Inventory": inventory
        }
    except Exception as e:
        return f"Error reading player data: {e}"

def get_servers_info(servers_dat_path):
    """
    Reads the list of multiplayer servers from servers.dat.
    """
    if not os.path.exists(servers_dat_path):
        return f"Error: File {servers_dat_path} not found."

    try:
        # servers.dat is usually uncompressed NBT
        nbt = nbtlib.load(servers_dat_path)
        servers = []
        for s in nbt.get('servers', []):
            servers.append({
                "Name": str(s.get('name', 'Unknown')),
                "IP": str(s.get('ip', 'Unknown')),
                "AcceptTextures": bool(s.get('acceptTextures', 0))
            })
        return servers
    except Exception as e:
        return f"Error reading servers.dat: {e}"

def get_map_data(map_dat_path):
    """
    Reads map data from a map_*.dat file.
    """
    if not os.path.exists(map_dat_path):
        return f"Error: File {map_dat_path} not found."

    try:
        nbt = nbtlib.load(map_dat_path)
        data = nbt.get('data', {})
        
        # Map data structure
        info = {
            "Dimension": str(data.get('dimension', 'unknown')),
            "Scale": int(data.get('scale', 0)),
            "Center": (int(data.get('xCenter', 0)), int(data.get('zCenter', 0))),
            "Locked": bool(data.get('locked', 0)),
            "ColorsLength": len(data.get('colors', []))
        }
        return info
    except Exception as e:
        return f"Error reading map data: {e}"

def get_poi_data(path):
    """Reads Point of Interest data from a poi/*.mca file."""
    if not os.path.exists(path):
        return f"Error: File {path} not found."
    try:
        region = anvil.Region.from_file(path)
        all_pois = []
        for cz in range(32):
            for cx in range(32):
                try:
                    chunk_data = region.chunk_data(cx, cz)
                    if not chunk_data:
                        continue
                    sections = chunk_data.get('Sections', [])
                    for section in sections:
                        records = section.get('Records', [])
                        for record in records:
                            all_pois.append({
                                "type": str(record.get('type', 'unknown')),
                                "pos": list(record.get('pos', [])),
                                "free_tickets": int(record.get('free_tickets', 0))
                            })
                except:
                    continue
        return all_pois
    except Exception as e:
        return f"Error reading POI data: {e}"

def get_raids_info(path):
    """Reads raid information from raids.dat."""
    if not os.path.exists(path):
        return f"Error: File {path} not found."
    try:
        nbt_file = nbtlib.load(path)
        data = nbt_file.get('data', {})
        raids_list = data.get('raids', [])
        results = []
        for r in raids_list:
            results.append({
                "id": int(r.get('id', 0)),
                "status": str(r.get('status', 'unknown')),
                "center": [int(x) for x in r.get('center', [])]
            })
        return results
    except Exception as e:
        return f"Error reading raids: {e}"

def get_advancements(path):
    """Reads player advancements from a .json file."""
    if not os.path.exists(path):
        return f"Error: File {path} not found."
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        return f"Error reading advancements: {e}"

def get_player_stats(path):
    """Reads player statistics from a .json file."""
    if not os.path.exists(path):
        return f"Error: File {path} not found."
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        return data.get('stats', {})
    except Exception as e:
        return f"Error reading stats: {e}"

def get_usercache(path):
    """Reads the user cache from usercache.json."""
    if not os.path.exists(path):
        return f"Error: File {path} not found."
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        return f"Error reading usercache: {e}"

def get_idcounts(path):
    """Reads map ID counts from idcounts.dat."""
    if not os.path.exists(path):
        return f"Error: File {path} not found."
    try:
        nbt_file = nbtlib.load(path)
        # idcounts.dat usually contains a flat list of tags like 'map', 'village', etc.
        return {k: int(v) for k, v in nbt_file.items()}
    except Exception as e:
        return f"Error reading idcounts: {e}"

def get_all_blocks_in_chunk(region_file, cx, cz):
    """
    Returns ALL blocks in a chunk as a dictionary.
    Returns dict with keys (x, y, z) -> block_string like 'minecraft:stone'
    This is needed for diff comparison.
    """
    if not os.path.exists(region_file):
        return f"Error: File {region_file} not found."

    try:
        region = anvil.Region.from_file(region_file)
        chunk = anvil.Chunk.from_region(region, cx, cz)
        
        blocks = {}
        # Iterate through all sections
        for section in chunk.sections:
            if section is None:
                continue
            
            # Global Y for this section
            y_base = section.y * 16
            
            # Check if section has blocks
            if not hasattr(section, 'palette') or not section.palette:
                continue
            
            # Get all blocks in this 16x16x16 section
            for y in range(16):
                for z in range(16):
                    for x in range(16):
                        try:
                            block = section.get_block(x, y, z)
                            block_name = f"{block.namespace}:{block.id}" if hasattr(block, 'namespace') else f"minecraft:{block.id}"
                            # Only store non-air blocks to save space
                            if block.id != 'air':
                                blocks[(x, y_base + y, z)] = block_name
                        except:
                            continue
        
        return blocks
    except Exception as e:
        return f"Error reading all blocks: {e}"

def compare_chunks(original_blocks, modified_blocks):
    """
    Compare two chunk block dictionaries and find differences.
    Returns list of (x, y, z, block_name, action) where action is 'add', 'remove', or 'change'
    """
    if isinstance(original_blocks, str):  # Error message
        return original_blocks
    if isinstance(modified_blocks, str):
        return modified_blocks
    
    diffs = []
    
    # Find added and changed blocks (in modified but not in original, or different)
    for pos, block in modified_blocks.items():
        if pos not in original_blocks:
            diffs.append((pos[0], pos[1], pos[2], block, 'add'))
        elif original_blocks[pos] != block:
            diffs.append((pos[0], pos[1], pos[2], block, 'change'))
    
    # Find removed blocks (in original but not in modified)
    for pos, block in original_blocks.items():
        if pos not in modified_blocks:
            diffs.append((pos[0], pos[1], pos[2], 'minecraft:air', 'remove'))
    
    return sorted(diffs, key=lambda x: (x[1], x[0], x[2]))  # Sort by y, then x, then z

def generate_setblock_commands(diffs, chunk_cx, chunk_cz, offset_x=0, offset_y=0, offset_z=0):
    """
    Generate Minecraft /setblock commands from chunk differences.
    chunk_cx, chunk_cz: chunk coordinates
    offset_x, offset_y, offset_z: global offset to apply to coordinates
    Returns list of commands.
    """
    commands = []
    
    for x, y, z, block, action in diffs:
        if action == 'remove':
            # For remove, we set to air
            block = 'minecraft:air'
        
        # Convert chunk-relative coords to global coords
        global_x = chunk_cx * 16 + x + offset_x
        global_y = y + offset_y
        global_z = chunk_cz * 16 + z + offset_z
        
        cmd = f"setblock {global_x} {global_y} {global_z} {block}"
        commands.append(cmd)
    
    return commands

def generate_fill_commands(diffs, chunk_cx, chunk_cz, offset_x=0, offset_y=0, offset_z=0):
    """
    Generate Minecraft /fill commands from chunk differences.
    Groups adjacent blocks of the same type into fill commands.
    Returns list of commands.
    """
    if isinstance(diffs, str):  # Error message
        return [diffs]
    
    # Group blocks by type
    blocks_by_type = {}
    for x, y, z, block, action in diffs:
        if action == 'remove':
            block = 'minecraft:air'
        if block not in blocks_by_type:
            blocks_by_type[block] = []
        blocks_by_type[block].append((x, y, z))
    
    commands = []
    
    for block_type, positions in blocks_by_type.items():
        if block_type == 'minecraft:air':
            # Use /fill for air (clearing)
            if positions:
                xs = [p[0] for p in positions]
                ys = [p[1] for p in positions]
                zs = [p[2] for p in positions]
                x1, x2 = min(xs), max(xs)
                y1, y2 = min(ys), max(ys)
                z1, z2 = min(zs), max(zs)
                
                global_x1 = chunk_cx * 16 + x1 + offset_x
                global_x2 = chunk_cx * 16 + x2 + offset_x
                global_y1 = y1 + offset_y
                global_y2 = y2 + offset_y
                global_z1 = chunk_cz * 16 + z1 + offset_z
                global_z2 = chunk_cz * 16 + z2 + offset_z
                
                cmd = f"fill {global_x1} {global_y1} {global_z1} {global_x2} {global_y2} {global_z2} {block_type}"
                commands.append(cmd)
        else:
            # For non-air blocks, use individual setblock commands
            # (fill would replace too many natural blocks)
            for x, y, z in positions:
                global_x = chunk_cx * 16 + x + offset_x
                global_y = y + offset_y
                global_z = chunk_cz * 16 + z + offset_z
                cmd = f"setblock {global_x} {global_y} {global_z} {block_type}"
                commands.append(cmd)
    
    return commands

def compare_worlds(original_dir, modified_dir, dimension="overworld", chunk_coords=None):
    """
    Compare two world directories and find player-built changes.
    
    Args:
        original_dir: Path to original world (e.g., fresh world from same seed)
        modified_dir: Path to modified world (player-built structures)
        dimension: Dimension to compare ('overworld', 'nether', 'end')
        chunk_coords: Optional list of (cx, cz) to compare specific chunks
    
    Returns:
        Dictionary with diff information and commands
    """
    dim_paths = {
        "overworld": "region",
        "nether": os.path.join("DIM-1", "region"),
        "end": os.path.join("DIM1", "region")
    }
    
    rel_path = dim_paths.get(dimension.lower())
    if not rel_path:
        return {"error": f"Unknown dimension: {dimension}"}
    
    orig_region_dir = os.path.join(original_dir, rel_path)
    mod_region_dir = os.path.join(modified_dir, rel_path)
    
    if not os.path.exists(orig_region_dir):
        return {"error": f"Original region dir not found: {orig_region_dir}"}
    if not os.path.exists(mod_region_dir):
        return {"error": f"Modified region dir not found: {mod_region_dir}"}
    
    # Get list of region files
    orig_regions = sorted([f for f in os.listdir(orig_region_dir) if f.endswith('.mca')])
    mod_regions = sorted([f for f in os.listdir(mod_region_dir) if f.endswith('.mca')])
    
    all_diffs = []
    summary = {"total_changes": 0, "chunks_compared": 0, "regions_compared": 0}
    
    # Compare each region file
    for region_file in orig_regions:
        if region_file not in mod_regions:
            continue
        
        orig_region_path = os.path.join(orig_region_dir, region_file)
        mod_region_path = os.path.join(mod_region_dir, region_file)
        
        # Extract region coords from filename (r.x.z.mca)
        parts = region_file[:-4].split('.')
        if len(parts) >= 3:
            rz = int(parts[1])
            rx = int(parts[2])
        else:
            continue
        
        summary["regions_compared"] += 1
        
        try:
            orig_region = anvil.Region.from_file(orig_region_path)
            mod_region = anvil.Region.from_file(mod_region_path)
            
            # Compare chunks in this region
            for cz in range(32):
                for cx in range(32):
                    # Check if chunk exists in both
                    orig_offset, _ = orig_region.chunk_location(cx, cz)
                    mod_offset, _ = mod_region.chunk_location(cx, cz)
                    
                    if orig_offset == 0 or mod_offset == 0:
                        continue
                    
                    # If specific chunks requested, skip others
                    if chunk_coords and (rx * 32 + cx, rz * 32 + cz) not in chunk_coords:
                        continue
                    
                    # Get all blocks from both chunks
                    orig_blocks = get_all_blocks_in_chunk(orig_region_path, cx, cz)
                    mod_blocks = get_all_blocks_in_chunk(mod_region_path, cx, cz)
                    
                    if isinstance(orig_blocks, str):
                        continue
                    if isinstance(mod_blocks, str):
                        continue
                    
                    # Compare
                    diffs = compare_chunks(orig_blocks, mod_blocks)
                    
                    if diffs:
                        global_cx = rx * 32 + cx
                        global_cz = rz * 32 + cz
                        all_diffs.append({
                            "chunk": (global_cx, global_cz),
                            "local_chunk": (cx, cz),
                            "diffs": diffs
                        })
                        summary["total_changes"] += len(diffs)
                        summary["chunks_compared"] += 1
                        
        except Exception as e:
            continue
    
    return {
        "summary": summary,
        "diffs": all_diffs
    }

def generate_rebuild_commands(diff_result, output_format="setblock"):
    """
    Generate commands to rebuild the world from diff result.
    
    Args:
        diff_result: Result from compare_worlds()
        output_format: 'setblock' (individual commands) or 'fill' (optimized fills)
    
    Returns:
        List of Minecraft commands
    """
    if "error" in diff_result:
        return [f"Error: {diff_result['error']}"]
    
    commands = []
    commands.append("# Minecraft World Rebuild Commands")
    commands.append(f"# Total changes: {diff_result['summary']['total_changes']}")
    commands.append(f"# Chunks modified: {diff_result['summary']['chunks_compared']}")
    commands.append("")
    
    for chunk_diff in diff_result["diffs"]:
        cx, cz = chunk_diff["local_chunk"]
        diffs = chunk_diff["diffs"]
        
        commands.append(f"# Chunk ({chunk_diff['chunk'][0]}, {chunk_diff['chunk'][1]})")
        
        if output_format == "fill":
            cmds = generate_fill_commands(diffs, cx, cz)
        else:
            cmds = generate_setblock_commands(diffs, cx, cz)
        
        commands.extend(cmds)
        commands.append("")
    
    return commands

def export_commands_to_file(commands, output_path):
    """Export commands to a .mcfunction file or shell script."""
    try:
        with open(output_path, 'w') as f:
            f.write('\n'.join(commands))
        return f"Commands exported to {output_path}"
    except Exception as e:
        return f"Error exporting: {e}"

def list_biomes_in_chunk(region_file, cx, cz):
    """
    Lists all unique biomes in a specific chunk.
    """
    if not os.path.exists(region_file):
        return f"Error: File {region_file} not found."

    try:
        region = anvil.Region.from_file(region_file)
        chunk = anvil.Chunk.from_region(region, cx, cz)
        
        biomes = set()
        # Biomes are 4x4x4 in 1.18+. 
        # We sample at intervals of 4 to be efficient.
        for y in range(-64, 320, 16): # Sample every section height
            for z in range(0, 16, 4):
                for x in range(0, 16, 4):
                    try:
                        b = chunk.get_biome(x, y, z)
                        # anvil-parser2 biome might be an object or string
                        name = getattr(b, 'name', str(b))
                        biomes.add(name)
                    except:
                        continue
        return sorted(list(biomes))
    except Exception as e:
        return f"Error reading biomes: {e}"

def find_blocks_in_region(region_file, block_id):
    """
    Finds all coordinates of a specific block ID within an entire region file.
    Returns list of (cx, cz, lx, ly, lz).
    """
    if not os.path.exists(region_file):
        return f"Error: File {region_file} not found."

    try:
        region = anvil.Region.from_file(region_file)
        all_found = []
        for cz in range(32):
            for cx in range(32):
                offset, _ = region.chunk_location(cx, cz)
                if offset > 0:
                    found = find_blocks_in_chunk(region_file, cx, cz, block_id)
                    if isinstance(found, list):
                        for lx, ly, lz in found:
                            all_found.append((cx, cz, lx, ly, lz))
        return all_found
    except Exception as e:
        return f"Error searching region: {e}"

def get_scoreboard_data(path):
    """Reads scoreboard data from scoreboard.dat."""
    if not os.path.exists(path):
        return f"Error: File {path} not found."
    try:
        nbt_file = nbtlib.load(path)
        data = nbt_file.get('data', {})
        
        objectives = []
        for obj in data.get('Objectives', []):
            objectives.append({
                "Name": str(obj.get('Name')),
                "CriteriaName": str(obj.get('CriteriaName')),
                "DisplayName": str(obj.get('DisplayName'))
            })
            
        scores = []
        for score in data.get('PlayerScores', []):
            scores.append({
                "Name": str(score.get('Name')),
                "Objective": str(score.get('Objective')),
                "Score": int(score.get('Score'))
            })
            
        return {
            "Objectives": objectives,
            "Scores": scores
        }
    except Exception as e:
        return f"Error reading scoreboard: {e}"

def get_structure_data(path):
    """Reads structure data from a .nbt file."""
    if not os.path.exists(path):
        return f"Error: File {path} not found."
    try:
        nbt_file = nbtlib.load(path)
        size = [int(x) for x in nbt_file.get('size', [])]
        palette = [str(b.get('Name')) for b in nbt_file.get('palette', [])]
        blocks = nbt_file.get('blocks', [])
        entities = nbt_file.get('entities', [])
        
        return {
            "Size": size,
            "PaletteSize": len(palette),
            "BlockCount": len(blocks),
            "EntityCount": len(entities),
            "Palette": palette[:10] # Sample
        }
    except Exception as e:
        return f"Error reading structure: {e}"

def get_entities_from_mca(path):
    """Reads entities from an entities/*.mca file."""
    if not os.path.exists(path):
        return f"Error: File {path} not found."
    try:
        region = anvil.Region.from_file(path)
        all_entities = []
        for cz in range(32):
            for cx in range(32):
                try:
                    # For entities/*.mca, the chunk data structure is different
                    # It usually has a top-level 'Entities' tag
                    chunk_data = region.chunk_data(cx, cz)
                    if not chunk_data:
                        continue
                    
                    entities = chunk_data.get('Entities', [])
                    for ent in entities:
                        all_entities.append({
                            "id": str(ent.get('id', 'unknown')),
                            "Pos": [float(x) for x in ent.get('Pos', [])],
                            "Health": float(ent.get('Health', 0.0))
                        })
                except:
                    continue
        return all_entities
    except Exception as e:
        return f"Error reading entities MCA: {e}"

def list_regions(world_path, dimension="overworld"):
    """Lists all region files in a world directory for a specific dimension."""
    paths = {
        "overworld": "region",
        "nether": os.path.join("DIM-1", "region"),
        "end": os.path.join("DIM1", "region")
    }
    
    rel_path = paths.get(dimension.lower())
    if not rel_path:
        return f"Error: Unknown dimension {dimension}"
        
    region_dir = os.path.join(world_path, rel_path)
    if not os.path.exists(region_dir):
        return []
    
    files = [f for f in os.listdir(region_dir) if f.endswith(".mca")]
    return sorted(files)

def summarize_world(world_path):
    """Summarizes world statistics across all dimensions."""
    dimensions = ["overworld", "nether", "end"]
    summary = {
        "WorldName": os.path.basename(os.path.abspath(world_path)),
        "Dimensions": {}
    }
    
    total_regions = 0
    
    for dim in dimensions:
        regions = list_regions(world_path, dim)
        if isinstance(regions, list):
            total_regions += len(regions)
            summary["Dimensions"][dim] = {
                "RegionFiles": len(regions)
            }
            
    summary["TotalRegionFiles"] = total_regions
    
    # Check for icon
    icon_path = os.path.join(world_path, "icon.png")
    summary["HasIcon"] = os.path.exists(icon_path)
    
    return summary

def get_world_uid(world_path):
    """Reads the world UUID from uid.dat if it exists."""
    uid_path = os.path.join(world_path, "uid.dat")
    if not os.path.exists(uid_path):
        return "None"
    try:
        with open(uid_path, "rb") as f:
            data = f.read()
            return data.hex()
    except Exception as e:
        return f"Error reading uid.dat: {e}"

def main():
    if len(sys.argv) < 3:
        print("Usage:")
        print("  python mc_reader.py <file.mca> get <x> <y> <z>")
        print("  python mc_reader.py <file.mca> list <cx> <cz>")
        print("  python mc_reader.py <file.mca> entities <cx> <cz>")
        print("  python mc_reader.py <file.mca> find <cx> <cz> <block_id>")
        print("  python mc_reader.py <file.mca> biome <x> <y> <z>")
        print("  python mc_reader.py <file.mca> chunks")
        print("  python mc_reader.py <level.dat> info")
        print("  python mc_reader.py <uuid.dat> player")
        print("  python mc_reader.py <servers.dat> servers")
        print("  python mc_reader.py <map_*.dat> map")
        print("  python mc_reader.py <poi/r.x.z.mca> poi")
        print("  python mc_reader.py <raids.dat> raids")
        print("  python mc_reader.py <uuid.json> advancements")
        print("  python mc_reader.py <uuid.json> stats")
        print("  python mc_reader.py <usercache.json> cache")
        print("  python mc_reader.py <idcounts.dat> idcounts")
        print("  python mc_reader.py <file.mca> list_biomes <cx> <cz>")
        print("  python mc_reader.py <file.mca> find_region <block_id>")
        print("  python mc_reader.py <world_dir> list_regions")
        print("  python mc_reader.py <world_dir> summarize")
        return

    mca_file = sys.argv[1]
    command = sys.argv[2]

    if command == "get":
        x, y, z = map(int, sys.argv[3:6])
        print(f"Block at ({x}, {y}, {z}): {get_block_at(mca_file, x, y, z)}")
    elif command == "list":
        cx, cz = map(int, sys.argv[3:5])
        blocks = list_blocks_in_chunk(mca_file, cx, cz)
        print(f"Unique blocks in chunk ({cx}, {cz}):")
        for b in blocks:
            print(f" - {b}")
    elif command == "entities":
        cx, cz = map(int, sys.argv[3:5])
        entities = list_entities_in_chunk(mca_file, cx, cz)
        print(f"Block entities in chunk ({cx}, {cz}):")
        for e in entities:
            print(f" - {e}")
    elif command == "find":
        cx, cz = map(int, sys.argv[3:5])
        block_id = sys.argv[5]
        coords = find_blocks_in_chunk(mca_file, cx, cz, block_id)
        print(f"Found '{block_id}' at these chunk-relative coords in ({cx}, {cz}):")
        for c in coords:
            print(f" - {c}")
    elif command == "biome":
        x, y, z = map(int, sys.argv[3:6])
        biome = get_biome_at(mca_file, x, y, z)
        print(f"Biome at ({x}, {y}, {z}): {biome}")
    elif command == "info":
        # For info, the first arg is level.dat instead of .mca
        info = get_world_info(mca_file)
        if isinstance(info, dict):
            print(f"World Information for {mca_file}:")
            for k, v in info.items():
                print(f"  {k}: {v}")
        else:
            print(info)
    elif command == "chunks":
        chunks = list_chunks_in_region(mca_file)
        if isinstance(chunks, list):
            print(f"Generated chunks in {mca_file} ({len(chunks)} total):")
            for cx, cz in chunks:
                print(f"  ({cx}, {cz})")
        else:
            print(chunks)
    elif command == "player":
        data = get_player_data(mca_file)
        if isinstance(data, dict):
            print(f"Player Data for {mca_file}:")
            print(f"  Position: {data['Position']}")
            print(f"  Dimension: {data['Dimension']}")
            print(f"  XP Level: {data['XP Level']}")
            print(f"  Inventory:")
            for item in data['Inventory']:
                print(f"    - {item}")
        else:
            print(data)
    elif command == "servers":
        servers = get_servers_info(mca_file)
        if isinstance(servers, list):
            print(f"Multiplayer Servers in {mca_file}:")
            for s in servers:
                print(f"  - {s['Name']} ({s['IP']}) [Textures: {s['AcceptTextures']}]")
        else:
            print(servers)
    elif command == "map":
        info = get_map_data(mca_file)
        if isinstance(info, dict):
            print(f"Map Data for {mca_file}:")
            for k, v in info.items():
                print(f"  {k}: {v}")
        else:
            print(info)
    elif command == "poi":
        pois = get_poi_data(mca_file)
        if isinstance(pois, list):
            print(f"Points of Interest in {mca_file}: {len(pois)} found")
            for p in pois[:20]: # Limit output
                print(f"  - {p['type']} at {p['pos']} (Tickets: {p['free_tickets']})")
            if len(pois) > 20:
                print(f"  ... and {len(pois)-20} more")
        else:
            print(pois)
    elif command == "raids":
        raids = get_raids_info(mca_file)
        if isinstance(raids, list):
            print(f"Raids in {mca_file}:")
            for r in raids:
                print(f"  - ID: {r.get('id')}, Status: {r.get('status')}, Center: {r.get('center')}")
        else:
            print(raids)
    elif command == "advancements":
        adv = get_advancements(mca_file)
        if isinstance(adv, dict):
            print(f"Advancements in {mca_file}:")
            for k, v in adv.items():
                done = "Done" if v.get('done') else "In Progress"
                print(f"  - {k}: {done}")
        else:
            print(adv)
    elif command == "stats":
        stats = get_player_stats(mca_file)
        if isinstance(stats, dict):
            print(f"Player Statistics for {mca_file}:")
            for category, values in stats.items():
                print(f"  Category: {category}")
                for k, v in list(values.items())[:10]: # Limit output
                    print(f"    - {k}: {v}")
                if len(values) > 10:
                    print(f"    ... and {len(values)-10} more")
        else:
            print(stats)
    elif command == "cache":
        cache = get_usercache(mca_file)
        if isinstance(cache, list):
            print(f"User Cache in {mca_file}:")
            for entry in cache:
                print(f"  - {entry.get('name')} ({entry.get('uuid')}) [Expires: {entry.get('expiresOn')}]")
        else:
            print(cache)
    elif command == "idcounts":
        counts = get_idcounts(mca_file)
        if isinstance(counts, dict):
            print(f"ID Counts for {mca_file}:")
            for k, v in counts.items():
                print(f"  {k}: {v}")
        else:
            print(counts)
    elif command == "scoreboard":
        sb = get_scoreboard_data(mca_file)
        if isinstance(sb, dict):
            print(f"Scoreboard Data for {mca_file}:")
            print(f"  Objectives: {len(sb['Objectives'])}")
            for obj in sb['Objectives']:
                print(f"    - {obj['DisplayName']} ({obj['Name']})")
            print(f"  Scores: {len(sb['Scores'])}")
            for score in sb['Scores'][:10]:
                print(f"    - {score['Name']}: {score['Score']} ({score['Objective']})")
        else:
            print(sb)
    elif command == "structure":
        struct = get_structure_data(mca_file)
        if isinstance(struct, dict):
            print(f"Structure Data for {mca_file}:")
            print(f"  Size: {struct['Size']}")
            print(f"  Blocks: {struct['BlockCount']}, Entities: {struct['EntityCount']}")
            print(f"  Palette Sample: {', '.join(struct['Palette'])}")
        else:
            print(struct)
    elif command == "entities_mca":
        ents = get_entities_from_mca(mca_file)
        if isinstance(ents, list):
            print(f"Entities in {mca_file}: {len(ents)} found")
            for e in ents[:10]:
                print(f"  - {e['id']} at {e['Pos']} (Health: {e['Health']})")
        else:
            print(ents)
    elif command == "list_biomes":
        cx, cz = map(int, sys.argv[3:5])
        biomes = list_biomes_in_chunk(mca_file, cx, cz)
        if isinstance(biomes, list):
            print(f"Biomes in chunk ({cx}, {cz}): {', '.join(biomes)}")
        else:
            print(biomes)
    elif command == "find_region":
        block_id = sys.argv[3]
        results = find_blocks_in_region(mca_file, block_id)
        if isinstance(results, list):
            print(f"Found '{block_id}' at {len(results)} locations in {mca_file}")
            for cx, cz, lx, ly, lz in results[:20]:
                print(f"  - Chunk ({cx}, {cz}) at local ({lx}, {ly}, {lz})")
            if len(results) > 20:
                print(f"  ... and {len(results)-20} more")
        else:
            print(results)
    elif command == "list_regions":
        dim = sys.argv[3] if len(sys.argv) > 3 else "overworld"
        regions = list_regions(mca_file, dim)
        if isinstance(regions, list):
            print(f"Region files in {mca_file} ({dim}):")
            for r in regions:
                print(f"  - {r}")
        else:
            print(regions)
    elif command == "summarize":
        summary = summarize_world(mca_file)
        if isinstance(summary, dict):
            print(f"World Summary for {summary['WorldName']}:")
            print(f"  Total Region Files: {summary['TotalRegionFiles']}")
            print(f"  Has Icon: {summary['HasIcon']}")
            for dim, data in summary['Dimensions'].items():
                print(f"  Dimension {dim}: {data['RegionFiles']} regions")
        else:
            print(summary)
    elif command == "uid":
        print(f"World UID: {get_world_uid(mca_file)}")
    elif command == "datapacks":
        packs = list_datapacks(mca_file)
        if isinstance(packs, list):
            print(f"Datapacks in {mca_file}: {len(packs)}")
            for p in packs:
                print(f"  - {p}")
        else:
            print(packs)
    elif command == "villages":
        dim = sys.argv[3] if len(sys.argv) > 3 else "overworld"
        vills = get_villages_data(mca_file, dim)
        if isinstance(vills, list):
            print(f"Villages in {mca_file} ({dim}): {len(vills)}")
            for v in vills:
                print(f"  - Center: {v['Center']}, Radius: {v['Radius']}, Pop: {v['PopSize']}, Golems: {v['Golems']}")
        else:
            print(vills)
    elif command == "world_info":
        # Aggregate info
        info = get_level_info(mca_file)
        summary = summarize_world(mca_file)
        uid = get_world_uid(mca_file)
        packs = list_datapacks(mca_file)
        
        print(f"=== World Info: {mca_file} ===")
        if isinstance(info, dict):
            print(f"Name: {info['LevelName']}")
            print(f"Version: {info['VersionName']} (Data: {info['DataVersion']})")
            print(f"Seed: {info['RandomSeed']}")
            print(f"Spawn: {info['SpawnX']}, {info['SpawnY']}, {info['SpawnZ']}")
            print(f"Time: {info['Time']}, DayTime: {info['DayTime']}")
        else:
            print(f"Level Info Error: {info}")
        
        print(f"UID: {uid}")
        
        if isinstance(summary, dict):
            print(f"Total Regions: {summary['TotalRegionFiles']}")
            for dim, ddata in summary['Dimensions'].items():
                print(f"  - {dim}: {ddata['RegionFiles']} regions")
        
        if isinstance(packs, list):
            print(f"Datapacks ({len(packs)}): {', '.join(packs) if packs else 'None'}")
            
        print("==============================")
    else:
        print(f"Unknown command: {command}")

# ============================================================================
# TERRAIN FROM SEED - Compare generated terrain with current world
# ============================================================================

def generate_terrain_from_seed(seed, x, y, z):
    """
    Generate expected terrain block at coordinates based on world seed.
    Uses simplified terrain generation for comparison.
    """
    if isinstance(seed, str):
        seed = int(hashlib.md5(seed.encode()).hexdigest()[:8], 16)
    
    rng = random.Random(seed ^ (x * 31337) ^ (z * 1337) ^ y)
    
    # Simplified terrain layers based on Y level
    if y < -60:
        return {"name": "minecraft:deepslate"}
    elif y < 0:
        return {"name": "minecraft:stone"}
    elif y < 63:
        return {"name": "minecraft:dirt"}
    elif y < 70:
        return {"name": "minecraft:grass_block"}
    return {"name": "minecraft:air"}

def compare_player_builds(world_path, backup_path=None, seed=None):
    """
    Compare current world with either a backup or generated terrain from seed.
    Returns list of blocks that differ (player-built structures).
    """
    # If backup exists, use compare_worlds
    if backup_path and os.path.exists(backup_path):
        return compare_worlds(backup_path, world_path)
    
    # Otherwise, use seed-based terrain generation
    if seed is None:
        # Try to read seed from level.dat
        level_path = os.path.join(world_path, "level.dat")
        if os.path.exists(level_path):
            try:
                level = nbtlib.load(level_path)
                seed = level.get("Data", {}).get("RandomSeed", 0)
            except:
                seed = 0
    
    region_dir = os.path.join(world_path, "region")
    if not os.path.exists(region_dir):
        return {"error": "No region files found"}
    
    diffs = []
    
    for filename in os.listdir(region_dir):
        if not filename.startswith("r.") or not filename.endswith(".mca"):
            continue
        
        parts = filename.split(".")
        rx, rz = int(parts[1]), int(parts[2])
        region_path = os.path.join(region_dir, filename)
        
        try:
            region = anvil.Region.from_file(region_path)
            
            for cx in range(32):
                for cz in range(32):
                    try:
                        chunk = anvil.Chunk.from_region(region, cx, cz)
                        
                        # Sample blocks (every 4 blocks for performance)
                        for y in range(-64, 320, 4):
                            for bz in range(0, 16, 4):
                                for bx in range(0, 16, 4):
                                    try:
                                        block = chunk.get_block(bx, y, bz)
                                        if block and block.id() != "minecraft:air":
                                            # Convert to global coordinates
                                            gx = rx * 512 + cx * 16 + bx
                                            gz = rz * 512 + cz * 16 + bz
                                            
                                            # Get expected terrain
                                            expected = generate_terrain_from_seed(seed, gx, y, gz)
                                            
                                            if block.id() != expected["name"]:
                                                diffs.append({
                                                    "x": gx,
                                                    "y": y,
                                                    "z": gz,
                                                    "current": block.id(),
                                                    "expected": expected["name"]
                                                })
                                    except:
                                        pass
                    except:
                        pass
        except:
            pass
    
    return {
        "seed": seed,
        "total_changes": len(diffs),
        "diffs": diffs
    }

def export_player_builds(world_path, output_file, backup_path=None, seed=None):
    """
    Export player-built blocks as Minecraft commands (setblock).
    Generates a .mcfunction file that can recreate player builds.
    """
    result = compare_player_builds(world_path, backup_path, seed)
    
    if "error" in result:
        return result
    
    commands = [f"# Player builds detected: {result['total_changes']} blocks"]
    commands.append(f"# Seed: {result.get('seed', 'unknown')}")
    commands.append("")
    
    for d in result.get("diffs", []):
        commands.append(f"setblock {d['x']} {d['y']} {d['z']} {d['current']}")
    
    export_commands_to_file(commands, output_file)
    
    return {
        "status": "success",
        "output_file": output_file,
        "total_commands": len(commands) - 3,
        "seed": result.get("seed")
    }

def export_fill_commands(world_path, output_file, backup_path=None, seed=None):
    """
    Export player builds using fill commands for more efficient recreation.
    Groups adjacent blocks of same type into fill commands.
    """
    result = compare_player_builds(world_path, backup_path, seed)
    
    if "error" in result:
        return result
    
    diffs = result.get("diffs", [])
    
    # Group by block type
    blocks_by_type = {}
    for d in diffs:
        block_type = d["current"]
        if block_type not in blocks_by_type:
            blocks_by_type[block_type] = []
        blocks_by_type[block_type].append((d["x"], d["y"], d["z"]))
    
    commands = [f"# Player builds: {len(diffs)} blocks in {len(blocks_by_type)} block types"]
    commands.append(f"# Seed: {result.get('seed', 'unknown')}")
    commands.append("")
    
    # Generate fill commands for each block type
    for block_type, coords in blocks_by_type.items():
        commands.append(f"# {block_type}: {len(coords)} blocks")
        for x, y, z in coords:
            commands.append(f"setblock {x} {y} {z} {block_type}")
        commands.append("")
    
    export_commands_to_file(commands, output_file)
    
    return {
        "status": "success",
        "output_file": output_file,
        "total_blocks": len(diffs),
        "block_types": len(blocks_by_type)
    }

if __name__ == "__main__":
    main()
