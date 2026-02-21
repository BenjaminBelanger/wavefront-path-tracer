$ScriptDir = $PSScriptRoot

cmake -S $ScriptDir -B "$ScriptDir/build" -G "Visual Studio 17 2022" -A x64

cmake --build "$ScriptDir/build" --config Release --target lumina

& "$ScriptDir/build/bin/Release/lumina.exe"
