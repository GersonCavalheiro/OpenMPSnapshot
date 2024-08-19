#!/usr/bin/perl

use strict;
use warnings;

my $csv_file = "omp_atomic_commands.csv";
open CSV_FILE, ">", $csv_file or die "Failed to open CSV file: $!";

my $file_count = 0;
my $line_count = 0;

foreach my $file (<*./*.*>) {
  $file_count++;
  open FILE, "<", $file or die "Failed to open file: $file";

  my $current_line = 0;
  my $command = "";

  while (<FILE>) {
    chomp;
    $current_line++;

    if (/^#pragma omp atomic/) {
      $line_count++;

      if (/\{/) {
        $command = $_;
        while (<FILE>) {
          chomp;
          $current_line++;

          if (/\}/) {
            $command .= "\n$_\n";
            # No need for break here, loop ends naturally when } is found
          } else {
            $command .= "$_\n";
          }
        }
      } else {
        $command = $_ . "\n";
      }

      print CSV_FILE "$file@$current_line@$command\n";
    }
  }

  close FILE;
}

close CSV_FILE;

print "Total files scanned: $file_count\n";
print "Total atomic commands found: $line_count\n";

