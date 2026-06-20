#define MyAppName "AutoLabel"
#define MyAppVersion "0.1.0"
#define MyAppPublisher "AutoLabel"
#define MyAppExeName "AutoLabel.exe"

[Setup]
AppId={{C9A837AF-6910-41D7-A50B-D1949A78C798}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={localappdata}\Programs\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
OutputDir=..\dist-installer
OutputBaseFilename=AutoLabel-Setup
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
UninstallDisplayIcon={app}\{#MyAppExeName}
SetupLogging=yes

[Languages]
Name: "korean"; MessagesFile: "compiler:Languages\Korean.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "바탕 화면 바로 가기"; GroupDescription: "추가 아이콘:"; Flags: unchecked

[Files]
Source: "..\dist\AutoLabel\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "..\README.md"; DestDir: "{app}\docs"; Flags: ignoreversion
Source: "..\transform_label_format.md"; DestDir: "{app}\docs"; Flags: ignoreversion
Source: "..\generate_label.md"; DestDir: "{app}\docs"; Flags: ignoreversion
Source: "..\configs\plugins.example.json"; DestDir: "{app}\configs"; Flags: ignoreversion

[Icons]
Name: "{autoprograms}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{#MyAppName} 실행"; Flags: nowait postinstall skipifsilent

