while (<>) {
  # Split the line into an array based on '/' delimiters
  @fields = split("/", $_);

  # Extract the desired content (between second and third '/')
  my $value = @fields[2];

  # Increment the count for the current value in the hash
  $unique_values{$value}++;

  # Print the extracted content (optional)
  # print $value . "\n";
}

# Print the unique values and their counts
print "\nUnique values and their counts:\n";
foreach (keys %unique_values) {
  print "$_: $unique_values{$_}\n";
}
