# Spot-check: generated answers on held-out-tool prompts

The first 6 held-out prompts, greedy, seed 0 of every condition. Tools offered are listed; none occurs in calibration data.

## 1. “How bad is the no2 level in Toronto today?”

offered: track_package, create_support_ticket, get_air_quality, reserve_table, convert_units, split_bill, play_playlist, get_showtimes · round 2 of 2

gold: `{"name": "get_air_quality", "arguments": {"city": "Toronto", "pollutant": "no2"}}`

| condition | output | category |
|---|---|---|
| fp | `<tool_call> {"name": "get_air_quality", "arguments": {"city": "Toronto", "pollutant": "no2"}} </tool_call>` | correct |
| rtn | `<tool_call> {"name": "get_air_quality", "arguments": {"city": "Toronto", "pollutant": "no2"}} </tool_call>` | correct |
| awq_short | `<tool_call> {"name": "get_air_quality", "arguments": {"city": "Toronto", "pollutant": "no2"}} </tool_call>` | correct |
| awq_long_nomask | `<tool_call> {"name": "get_air_quality", "arguments": {"city": "Toronto", "pollutant": "no2"}} </tool_call>` | correct |
| awq_long_masked | `<tool_call> {"name": "get_air_quality", "arguments": {"city": "Toronto", "pollutant": "no2"}} </tool_call>` | correct |

## 2. “Start Chill Jazz with shuffle off.”

offered: track_package, get_route_eta, get_showtimes, convert_units, reserve_table, split_bill, play_playlist · round 1 of 1

gold: `{"name": "play_playlist", "arguments": {"playlist": "Chill Jazz", "shuffle": false}}`

| condition | output | category |
|---|---|---|
| fp | `<tool_call> {"name": "play_playlist", "arguments": {"playlist": "Chill Jazz", "shuffle": true}} </tool_call>` | wrong_arg_value |
| rtn | `<tool_call> {"name": "play_playlist", "arguments": {"playlist": "Chill Jazz", "shuffle": true}} </tool_call>` | wrong_arg_value |
| awq_short | `<tool_call> {"name": "play_playlist", "arguments": {"playlist": "Chill Jazz", "shuffle": true}} </tool_call>` | wrong_arg_value |
| awq_long_nomask | `<tool_call> {"name": "play_playlist", "arguments": {"playlist": "Chill Jazz", "shuffle": true}} </tool_call>` | wrong_arg_value |
| awq_long_masked | `<tool_call> {"name": "play_playlist", "arguments": {"playlist": "Chill Jazz", "shuffle": true}} </tool_call>` | wrong_arg_value |

## 3. “How many SKU-5149 are left in YYZ1?”

offered: track_package, play_playlist, get_route_eta, get_showtimes, convert_units, check_inventory, get_air_quality · round 3 of 3

gold: `{"name": "check_inventory", "arguments": {"sku": "SKU-5149", "warehouse": "YYZ1"}}`

| condition | output | category |
|---|---|---|
| fp | `<tool_call> {"name": "get_air_quality", "arguments": {"city": "YYZ1", "pollutant": "pm25"}} </tool_call>` | wrong_tool |
| rtn | `<tool_call> {"name": "get_air_quality", "arguments": {"city": "YYZ1", "pollutant": "pm25"}} </tool_call>` | wrong_tool |
| awq_short | `<tool_call> {"name": "get_inventory", "arguments": {"sku": "SKU-5149", "warehouse": "YYZ1"}} </tool_call>` | hallucinated_tool |
| awq_long_nomask | `<tool_call> {"name": "get_inventory", "arguments": {"sku": "SKU-5149", "warehouse": "YYZ1"}} </tool_call>` | hallucinated_tool |
| awq_long_masked | `<tool_call> {"name": "get_inventory", "arguments": {"sku": "SKU-5149", "warehouse": "YYZ1"}} </tool_call>` | hallucinated_tool |

## 4. “How long from 697 Bloor St to 516 King St by driving?”

offered: get_route_eta, create_support_ticket, track_package, get_showtimes, convert_units · round 1 of 2

gold: `{"name": "get_route_eta", "arguments": {"origin": "697 Bloor St", "destination": "516 King St", "mode": "driving"}}`

| condition | output | category |
|---|---|---|
| fp | `<tool_call> {"name": "get_route_eta", "arguments": {"origin": "697 Bloor St", "destination": "516 King St", "mode": "driving"}} </tool_call>` | correct |
| rtn | `<tool_call> {"name": "get_route_eta", "arguments": {"origin": "697 Bloor St", "destination": "516 King St", "mode": "driving"}} </tool_call>` | correct |
| awq_short | `<tool_call> {"name": "get_route_eta", "arguments": {"origin": "697 Bloor St", "destination": "516 King St", "mode": "driving"}} </tool_call>` | correct |
| awq_long_nomask | `<tool_call> {"name": "get_route_eta", "arguments": {"origin": "697 Bloor St", "destination": "516 King St", "mode": "driving"}} </tool_call>` | correct |
| awq_long_masked | `<tool_call> {"name": "get_route_eta", "arguments": {"origin": "697 Bloor St", "destination": "516 King St", "mode": "driving"}} </tool_call>` | correct |

## 5. “Showtimes for The Long Walk in Halifax, 2026-05-13.”

offered: reserve_table, track_package, convert_units, get_showtimes, get_route_eta, play_playlist · round 2 of 3

gold: `{"name": "get_showtimes", "arguments": {"movie": "The Long Walk", "city": "Halifax", "date": "2026-05-13"}}`

| condition | output | category |
|---|---|---|
| fp | `<tool_call> {"name": "get_showtimes", "arguments": {"movie": "The Long Walk", "city": "Halifax, 2026-05-13"}} </tool_call>` | wrong_arg_keys |
| rtn | `<tool_call> {"name": "track_package", "arguments": {"tracking_number": "0001", "carrier": "UPS"}} </tool_call>` | wrong_tool |
| awq_short | `<tool_call> {"name": "track_package", "arguments": {"tracking_number": "123456789", "carrier": "UPS"}} </tool_call>` | wrong_tool |
| awq_long_nomask | `<tool_call> {"name": "track_package", "arguments": {"tracking_number": "00000000000000000000000000000000", "carrier": "UPS"}} </tool_call>` | wrong_tool |
| awq_long_masked | `<tool_call> {"name": "get_showtimes", "arguments": {"movie": "The Long Walk in Halifax", "city": "Halifax", "date": "2026-05-13"}} </tool_call>` | wrong_arg_value |

## 6. “Log this as urgent: wrong item delivered.”

offered: create_support_ticket, check_inventory, reserve_table, get_showtimes, track_package, get_route_eta, get_air_quality, split_bill · round 1 of 2

gold: `{"name": "create_support_ticket", "arguments": {"summary": "wrong item delivered", "priority": "urgent"}}`

| condition | output | category |
|---|---|---|
| fp | `<tool_call> {"name": "track_package", "arguments": {"tracking_number": "wrong_item_delivered", "carrier": "FedEx"}} </tool_call>` | wrong_tool |
| rtn | `I'm sorry, but I'm unable to assist with that yet. I need more information to understand what you're trying to achieve. Could you please provide more details about the issue and what you would like me to do about it?` | no_call |
| awq_short | `<tool_call> {"name": "log_issue", "arguments": {"description": "wrong item delivered"}} </tool_call>` | hallucinated_tool |
| awq_long_nomask | `<tool_call> {"name": "check_inventory", "arguments": {"sku": "wrong item", "warehouse": "YYZ1"}} </tool_call>` | wrong_tool |
| awq_long_masked | `<tool_call> {"name": "check_inventory", "arguments": {"sku": "wrong_item", "warehouse": "YYZ1"}} </tool_call>` | wrong_tool |

