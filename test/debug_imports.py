#!/usr/bin/env python3
import sys
import os
import asyncio
from pathlib import Path

print("🔧 Debugging Async LLM Query Module...")

# Get the root directory
ROOT_DIR = Path(__file__).parent.absolute()
print(f"📁 Root directory: {ROOT_DIR}")

# Add src to path
src_path = Path("/kayla/llm_query_module/src")
print(f"📂 Src directory: {src_path}")
print(f"📂 Src exists: {src_path.exists()}")

if src_path.exists():
    print("📄 Files in src/:")
    for file in src_path.iterdir():
        print(f"  - {file.name}")

sys.path.insert(0, str(src_path))
os.chdir(ROOT_DIR)

# Test imports one by one
print("\n🧪 Testing imports...")

try:
    print("  Importing async_main_interface...")
    from async_main_interface import AsyncLLMQueryInterface
    print("  ✅ async_main_interface imported successfully")
except Exception as e:
    print(f"  ❌ Failed to import async_main_interface: {e}")

# Test if AsyncLLMQueryInterface works asynchronously
async def test_async_import():
    try:
        print("  Testing AsyncLLMQueryInterface functionality...")
        config_path = "/kayla/llm_query_module/config/query_config.yaml"  # Adjust path if necessary
        interface = AsyncLLMQueryInterface(config_path)
        print("  ✅ AsyncLLMQueryInterface initialized successfully")

        # Test the interactive menu (optionally, you can choose specific methods)
        await interface.main_menu()
        
    except Exception as e:
        print(f"  ❌ Error during test: {e}")

# Run the async test
asyncio.run(test_async_import())

print("\n🎯 Debug complete!")
