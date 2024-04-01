past_key_values = (1, 2, 3, 4, 5, 6, 7, 8)
outputs = [9, 10, 11, 12, 13, 14, 15, 16]

out_past_key_values = tuple(outputs[key] for key in range(8))

print(out_past_key_values)
# out_past_key_values = tuple(
#     out_past_key_values[i : i + 2] + past_key_values[i + 2 : i + 4]
#     for i in range(0, len(out_past_key_values), 4)
# )

# print(out_past_key_values)

out_past_key_values = tuple(out_past_key_values[i : i + 4] for i in range(0, len(out_past_key_values), 4))
print(out_past_key_values)

past_key_values = ((1, 2), (3, 4), (5, 6), (7, 8))

past_key_values = tuple(past_key_value for pkv_per_layer in past_key_values for past_key_value in pkv_per_layer)

print(past_key_values)
