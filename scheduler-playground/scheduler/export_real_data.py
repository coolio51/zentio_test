#!/usr/bin/env python3
"""
Export real scheduler data to JSON files for testing.

This script fetches the actual data used by the scheduler from the zentio-v1 API
and saves it to JSON files that can be used for testing.
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Optional

import httpx

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)


async def resolve_org_id_by_slug(
    client: httpx.AsyncClient, server_url: str, slug: str
) -> str:
    """Resolve organization ID from stable slug via public tRPC endpoint."""
    resp = await client.get(f"{server_url}/trpc/public.organizations.getAll")
    resp.raise_for_status()
    data = resp.json()
    try:
        organizations = data["result"]["data"]["json"]
    except Exception:
        raise RuntimeError("Unexpected response format from organizations endpoint")

    for org in organizations:
        if org.get("slug") == slug:
            org_id = org.get("id")
            if not org_id:
                break
            return org_id
    raise RuntimeError(f"Organization with slug '{slug}' not found")


async def fetch_real_data(
    organization_id: Optional[str] = None,
    days_ahead: int = 120,
    organization_slug: str = "acme",
) -> Dict:
    """
    Fetch real data from the zentio-v1 server.

    Args:
        organization_id: The organization ID to fetch data for
        days_ahead: Number of days ahead to schedule (now using 2 months)
    """

    # Calculate date range (today + 2 months)
    start_date = datetime.now()
    end_date = start_date + timedelta(days=days_ahead)

    start_date_str = start_date.isoformat()
    end_date_str = end_date.isoformat()

    # Get the zentio-v1 server URL from environment or use default
    server_url = os.getenv("ZENTIO_V1_SERVER_URL", "http://localhost:8080")

    print(f"🔍 Fetching data from {server_url}")
    print(f"📅 Date range: {start_date_str} to {end_date_str}")
    async with httpx.AsyncClient() as client:
        try:
            # Resolve organization ID from slug if not provided or if ENV requests slug resolution
            org_id = organization_id or os.getenv("ORGANIZATION_ID")
            if not org_id:
                slug = os.getenv("ORGANIZATION_SLUG", organization_slug)
                org_id = await resolve_org_id_by_slug(client, server_url, slug)

            print(f"🏢 Organization: {org_id}")

            # Fetch manufacturing orders
            print("📦 Fetching manufacturing orders...")
            mo_response = await client.get(
                f"{server_url}/api/scheduler/manufacturing-orders",
                headers={
                    "X-Organization-ID": org_id,
                    "Content-Type": "application/json",
                },
                params={
                    "start_date": start_date_str,
                    "end_date": end_date_str,
                },
            )
            mo_response.raise_for_status()
            manufacturing_orders_data = mo_response.json()

            # Handle both list and dict formats
            if isinstance(manufacturing_orders_data, list):
                mo_count = len(manufacturing_orders_data)
            else:
                mo_count = len(
                    manufacturing_orders_data.get("manufacturingOrdersRequirements", [])
                )

            print(f"✅ Found {mo_count} manufacturing orders")

            # Fetch available resources
            print("🛠️  Fetching available resources...")
            resources_response = await client.get(
                f"{server_url}/api/scheduler/available-resources",
                headers={
                    "X-Organization-ID": org_id,
                    "Content-Type": "application/json",
                },
                params={
                    "start_date": start_date_str,
                    "end_date": end_date_str,
                },
            )
            resources_response.raise_for_status()
            resources_data = resources_response.json()

            # Handle both list and dict formats
            if isinstance(resources_data, list):
                resources_count = len(resources_data)
            else:
                resources_count = len(resources_data.get("resources", []))

            print(f"✅ Found {resources_count} resources")

            return {
                "manufacturing_orders": manufacturing_orders_data,
                "resources": resources_data,
                "metadata": {
                    "organization_id": org_id,
                    "start_date": start_date_str,
                    "end_date": end_date_str,
                    "exported_at": datetime.now().isoformat(),
                    "days_ahead": days_ahead,
                },
            }

        except httpx.HTTPStatusError as e:
            print(f"❌ HTTP error: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            print(f"❌ Error fetching data: {e}")
            raise


async def main():
    """Export real scheduler data to JSON files."""

    try:
        # Create data directory
        data_dir = Path("tests/data")
        data_dir.mkdir(parents=True, exist_ok=True)

        # Fetch real data (resolve org by stable slug 'acme' unless ORGANIZATION_ID is set)
        data = await fetch_real_data()

        # Save individual files
        mo_file = data_dir / "real_manufacturing_orders.json"
        resources_file = data_dir / "real_resources.json"
        metadata_file = data_dir / "real_metadata.json"
        combined_file = data_dir / "real_combined.json"

        print(f"💾 Saving data to {data_dir}...")

        # Save manufacturing orders
        with open(mo_file, "w") as f:
            json.dump(data["manufacturing_orders"], f, indent=2, default=str)
        print(f"✅ Saved manufacturing orders: {mo_file}")

        # Save resources
        with open(resources_file, "w") as f:
            json.dump(data["resources"], f, indent=2, default=str)
        print(f"✅ Saved resources: {resources_file}")

        # Save metadata
        with open(metadata_file, "w") as f:
            json.dump(data["metadata"], f, indent=2, default=str)
        print(f"✅ Saved metadata: {metadata_file}")

        # Save combined file
        with open(combined_file, "w") as f:
            json.dump(data, f, indent=2, default=str)
        print(f"✅ Saved combined data: {combined_file}")

        print("🎉 Data export complete!")
        print(f"📊 Summary:")

        # Handle both list and dict formats for summary
        mo_data = data["manufacturing_orders"]
        if isinstance(mo_data, list):
            mo_summary_count = len(mo_data)
        else:
            mo_summary_count = len(mo_data.get("manufacturingOrdersRequirements", []))

        resources_summary_data = data["resources"]
        if isinstance(resources_summary_data, list):
            resources_summary_count = len(resources_summary_data)
        else:
            resources_summary_count = len(resources_summary_data.get("resources", []))

        print(f"   • Manufacturing Orders: {mo_summary_count}")
        print(f"   • Resources: {resources_summary_count}")
        print(f"   • Date Range: {data['metadata']['days_ahead']} days ahead")

    except Exception as e:
        print(f"❌ Export failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
