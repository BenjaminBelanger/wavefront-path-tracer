$ScriptDir = $PSScriptRoot

cmake -S $ScriptDir -B "$ScriptDir/build" -G "Visual Studio 17 2022" -A x64

cmake --build "$ScriptDir/build" --config Release --target wavefront-path-tracer

#& "$ScriptDir/build/bin/Release/wavefront-path-tracer.exe"
