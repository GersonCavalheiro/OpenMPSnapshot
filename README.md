# OpenMP Repository Analysis

# Presentation
This repository accompanies the article:
A Snapshot of OpenMP Projects on GitHub

This content has been made available to assist in the reproduction of the research conducted and to support the development of new related works. When using this material, either as presented or modified, we request that the following article be cited as the source:

```
@inproceedings{OpenMPSnapshotSBLP2024,
author = {Void, Null},
title = {A Snapshot of OpenMP Projects on GitHub}, year = {2024},
publisher = {Association for Computing Machinery},
address = {Curitiba, Brazil},
note = {Submitted}
}
```
# Contents
- **May2023**: Contains the repositories mined in May 2023 used in the base article, divided into repositories with the primary language C and those with the primary language C++.
- **Python**: Script for mining GitHub repositories.
- **AWK**: Scripts for extracting content for the analyses documented in the article.
- **PartialData**: Files with temporary data generated during the mining and content extraction process.

# Utilization
This set of artifacts was developed in a Linux/Ubuntu 22.04 environment and uses Python 3.x. Before using it, make sure your system has Python installed in the indicated version and the PyGithub and Pandas libraries. You should also have a personal access token to GitHub (access the platform with your GitHub account to obtain it). The other tools used are the sed line editor and the AWK text manipulation language, which usually come with the installation of the Linux/Ubuntu environment. The grep command is also used.

## Step 1: Mining
This program searches for repositories on GitHub that use the OpenMP library in C and C++ languages. It uses the GitHub API to perform the search and extract relevant information from each repository.

1. Edit the `minner.py` file and insert your personal GitHub access token in the place indicated by `INSERT_YOUR_PERSONAL_ACCESS_TOKEN`.
2. Launch the mining program in a terminal:
   
    ```
    $ python3 minner.py
    ```
4. During the execution of the script, you will be asked if repositories with C or C++ as the main language should be mined.
5. Organize the produced directory and the resulting CSV file.

## Step 2: Analysis

### 1. Cleaning
1.1 Manually, and carefully, remove all files from the repositories that do not have one of the following extensions: `.c`, `.h`, `.C`, `.H`, `.cpp`, `.hpp`, `.cxx`, `.hxx`, `.inl`.

1.2 Remove leading spaces and tabs to simplify the identification of "#pragma omp directives" occurrences (run it as many times as necessary, test with `grep '\\$' May2023/RepC/*/*.*`):

    ```
    $ sed -i -E 's/^[ \t]+//' May2023/RepC/*/*.*
    $ sed -i -E 's/^[ \t]+//' May2023/RepCPP/*/*.*
    ```

1.3 Remove leading spaces and tabs from lines starting with "#" followed by the string "pragma":

    ```
    $ sed -i -E ':a; /^[[:space:]]*#pragma.*\\$/ { N; s/\\\n//; ba; }' May2023/RepC/*/*.*
    $ sed -i -E ':a; /^[[:space:]]*#pragma.*\\$/ { N; s/\\\n//; ba; }' May2023/RepCPP/*/*.*
    ```

### 2. Extract Data

2.1 Count directives:

    ```
    $ awk -f AWK/CountingDirectives.awk May2023/RepC/*/*.* > PartialData/CountRepC.csv
    $ awk -f AWK/CountingDirectives.awk May2023/RepCPP/*/*.* > PartialData/CountRepCPP.csv
    ```

2.2 Count atomic clauses:

    ```
    $ awk -f AWK/CountingClausesAtomic.awk May2023/RepC/*/*.* May2023/RepCPP/*/*.*
    ```

2.3 Count "Critical" with label:

    ```
    $ grep -E "^#pragma omp critical[ \t]*(\(.+\))" May2023/RepC*/*/*.* | wc -l
    ```

2.4 Count atomic critical candidates:

    ```
    $ awk -f AWK/casosCritical.awk May2023/RepC*/*/*.*
    ```

2.5 Count "schedule for":

    ```
    $ awk -f AWK/contaScheduleFor.awk May2023/RepC*/*/*.*
    ```

2.6 Count "for" without "schedule":

    ```
    $ grep -E "^#pragma omp parallel for" May2023/RepC*/*/*.* | grep -v schedule  | wc -l
    $ grep -E "^#pragma omp  for" May2023/RepC*/*/*.* | grep -v schedule  | wc -l
    ```

2.7 Count number of unbalanced loops:

    ```
    $ awk -f AWK/unbalancedLoop.awk May2023/RepC*/*/*.*
    ```

2.8 Identify blocks with a sequence of "parallel for":

    ```
    $ awk -f AWK/closeParallelFor.awk May2023/RepC*/*/*.* | wc -l
    ```

