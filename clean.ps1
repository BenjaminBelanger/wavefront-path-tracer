# Remove-Comments.ps1
# Removes C-style comments (// and /* */) from all files recursively
# Run this script from the root of your CUDA project

param(
    [string]$RootPath = $PSScriptRoot
)

Write-Host "Starting comment removal in: $RootPath" -ForegroundColor Cyan

$excludeDirs = @("build", ".git", "external")
$files = Get-ChildItem -Path $RootPath -Recurse -File | Where-Object {
    $rel = $_.FullName.Substring($RootPath.Length).TrimStart('\', '/')
    $skip = $false
    foreach ($d in $excludeDirs) {
        if ($rel -like "$d\*" -or $rel -like "$d/*") { $skip = $true; break }
    }
    -not $skip
}

foreach ($file in $files) {
    if ($file.FullName -eq $MyInvocation.MyCommand.Path) { continue }

    try {
        $content = Get-Content -Path $file.FullName -Raw -ErrorAction Stop
    } catch {
        Write-Warning "Could not read: $($file.FullName)"
        continue
    }

    if ($null -eq $content) { continue }

    $original = $content

    # --- Remove block comments /* ... */ (including multiline) ---
    $content = [regex]::Replace($content, '/\*[\s\S]*?\*/', '', 'None')

    # --- Remove line comments // ... (but NOT after http:) ---
    $content = [regex]::Replace($content, '(?<!:)//[^\r\n]*', '', 'None')

    if ($content -ne $original) {
        Set-Content -Path $file.FullName -Value $content -NoNewline
        Write-Host "Cleaned: $($file.FullName)" -ForegroundColor Green
    }
}

Write-Host "`nDone!" -ForegroundColor Cyan
