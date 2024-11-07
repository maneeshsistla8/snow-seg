#!/bin/bash

# Check for the correct number of arguments
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <start_date> <end_date> <geojson_prefix> <order_prefix>"
    exit 1
fi

# Extract command-line arguments
START_DATE=$1
END_DATE=$2
GEOJSON_PREFIX=$3
ORDER_PREFIX=$4

# Fetch the latest IDs based on the provided dates and geom from the specified geojson
LATEST_ID=$(planet data filter \
    --range cloud_percent lt 90 \
    --date-range acquired gte "$START_DATE" \
    --date-range acquired lt "$END_DATE" \
    --geom "${GEOJSON_PREFIX}.geojson" \
    --permission --std-quality | \
    planet data search PSScene \
    --sort 'acquired asc' \
    --limit 5000 \
    --filter - | \
    jq -r .id | \
    tr '\n' ',' | \
    sed 's/.$//')

total_scenes=$(echo "$LATEST_ID" | tr ',' '\n' | wc -l)
# Print the result
#echo "IDs: $LATEST_ID"
echo "Total Number: $total_scenes"

# Function to split IDs and create order requests
create_orders() {
    ids=$1
    prefix=$2
    chunk_size=500
    total_chunks=$(( (total_scenes + chunk_size - 1) / chunk_size ))

    for ((i=0; i<$total_chunks; i++)); do
        start=$((i * chunk_size + 1))
        end=$(( (i + 1) * chunk_size ))
        chunk=$(echo "$ids" | cut -d',' -f$start-$end)
        order_name="${prefix}_part$((i + 1))"
        planet orders request \
            --item-type PSScene \
            --bundle analytic_sr_udm2 \
            --clip "${GEOJSON_PREFIX}.geojson" \
            --name "$order_name" \
            $chunk \
            --archive-filename {{name}}.zip --archive-type zip \
            --email --single-archive > "$order_name.json"
        echo "Request file: $order_name.json"
    done
}

create_orders "$LATEST_ID" "$ORDER_PREFIX.$START_DATE.$END_DATE"