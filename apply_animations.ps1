# Apply professional, cohesive presentation animations to ReSonus_Presentation.pptx
param(
    [string]$pptxPath = "D:\audio_toolbox_desktop\ReSonus_Presentation.pptx"
)

Write-Host "Connecting to PowerPoint COM engine..."
$ppt = New-Object -ComObject PowerPoint.Application
# Open: FileName, ReadOnly=0, Untitled=0, WithWindow=0
$pres = $ppt.Presentations.Open($pptxPath, 0, 0, 0)

# PowerPoint Animation Enums
$EFFECT_FADE   = 10 # msoAnimEffectFade
$EFFECT_FLOAT  = 71 # msoAnimEffectRiseUp (Float In)
$EFFECT_WIPE   = 22 # msoAnimEffectWipe
$TRIG_CLICK    = 1  # msoAnimTriggerOnPageClick
$TRIG_WITH     = 2  # msoAnimTriggerWithPrevious
$TRIG_AFTER    = 3  # msoAnimTriggerAfterPrevious

function Animate-Group {
    param(
        $slide,
        [int[]]$shapeIndices,
        [int]$leadTrigger = $TRIG_AFTER,
        [int]$effect = $EFFECT_FADE,
        [float]$duration = 0.35,
        [float]$leadDelay = 0.0
    )
    if ($shapeIndices.Count -eq 0) { return }
    
    # Lead shape in the group
    $firstIdx = $shapeIndices[0]
    if ($firstIdx -le $slide.Shapes.Count) {
        $shpFirst = $slide.Shapes[$firstIdx]
        $effFirst = $slide.TimeLine.MainSequence.AddEffect($shpFirst, $effect, 0, $leadTrigger)
        $effFirst.Timing.Duration = $duration
        if ($leadDelay -gt 0) {
            $effFirst.Timing.TriggerDelayTime = $leadDelay
        }
    }
    
    # Remaining shapes in the group enter simultaneously (WITH PREVIOUS)
    for ($k = 1; $k -lt $shapeIndices.Count; $k++) {
        $idx = $shapeIndices[$k]
        if ($idx -le $slide.Shapes.Count) {
            $shp = $slide.Shapes[$idx]
            $eff = $slide.TimeLine.MainSequence.AddEffect($shp, $effect, 0, $TRIG_WITH)
            $eff.Timing.Duration = $duration
        }
    }
}

Write-Host "Configuring transitions and build animations for $($pres.Slides.Count) slides..."

for ($i = 1; $i -le $pres.Slides.Count; $i++) {
    $slide = $pres.Slides[$i]
    
    # Slide Transition: Smooth Fade (ppEffectFade = 3849)
    $slide.SlideShowTransition.EntryEffect = 3849
    $slide.SlideShowTransition.Duration = 0.5
    
    # Clear any existing animation nodes
    while ($slide.TimeLine.MainSequence.Count -gt 0) {
        $slide.TimeLine.MainSequence[1].Delete()
    }
    
    switch ($i) {
        1 {
            # SLIDE 1: Title & Hero
            # 5: "ENGINEERING PRESENTATION" badge
            # 6: Main Title & Subtitle
            # 7..11: 5 Module Pills (staggered cascade)
            # 12..14: 3 Metric Cards
            Animate-Group $slide @(5, 6) $TRIG_AFTER $EFFECT_FADE 0.45 0.1
            
            # Stagger module pills
            for ($p = 7; $p -le 11; $p++) {
                if ($p -le $slide.Shapes.Count) {
                    $eff = $slide.TimeLine.MainSequence.AddEffect($slide.Shapes[$p], $EFFECT_FADE, 0, $TRIG_AFTER)
                    $eff.Timing.Duration = 0.15
                }
            }
            
            # Bottom 3 metric cards enter together
            Animate-Group $slide @(12, 13, 14) $TRIG_AFTER $EFFECT_FADE 0.35 0.1
        }
        
        2 {
            # SLIDE 2: Architecture Matrix (5 Columns)
            # Header items are already on slide
            # Col 1: 7..11, Col 2: 12..16, Col 3: 17..21, Col 4: 22..26, Col 5: 27..31
            Animate-Group $slide @(7, 8, 9, 10, 11) $TRIG_AFTER $EFFECT_FADE 0.3 0.1
            Animate-Group $slide @(12, 13, 14, 15, 16) $TRIG_AFTER $EFFECT_FADE 0.3 0.05
            Animate-Group $slide @(17, 18, 19, 20, 21) $TRIG_AFTER $EFFECT_FADE 0.3 0.05
            Animate-Group $slide @(22, 23, 24, 25, 26) $TRIG_AFTER $EFFECT_FADE 0.3 0.05
            Animate-Group $slide @(27, 28, 29, 30, 31) $TRIG_AFTER $EFFECT_FADE 0.3 0.05
        }
        
        { $_ -in 3, 4, 5, 6, 9, 10, 11 } {
            # DSP Detail Slides (Noise Remover, Equalizer, Morse, Matcher)
            # Left Card (Concept & Steps): Shapes 7, 8, 9, 10
            # Right Card (DSP Chart & Takeaway): Shapes 11, 12, 13, 14
            Animate-Group $slide @(7, 8, 9, 10) $TRIG_AFTER $EFFECT_FADE 0.35 0.1
            Animate-Group $slide @(11, 12, 13, 14) $TRIG_AFTER $EFFECT_FADE 0.4 0.12
        }
        
        7 {
            # SLIDE 7: Audio Editor (Trim & Concatenate)
            # Left Card (Trimming): Shapes 7, 8, 9, 10
            # Right Card (Concatenation): Shapes 11, 12, 13, 14
            Animate-Group $slide @(7, 8, 9, 10) $TRIG_AFTER $EFFECT_FADE 0.35 0.1
            Animate-Group $slide @(11, 12, 13, 14) $TRIG_AFTER $EFFECT_FADE 0.35 0.12
        }
        
        8 {
            # SLIDE 8: Audio Editor (4 Quadrants)
            # Q1: 7..11, Q2: 12..16, Q3: 17..21, Q4: 22..26
            Animate-Group $slide @(7, 8, 9, 10, 11) $TRIG_AFTER $EFFECT_FADE 0.28 0.1
            Animate-Group $slide @(12, 13, 14, 15, 16) $TRIG_AFTER $EFFECT_FADE 0.28 0.06
            Animate-Group $slide @(17, 18, 19, 20, 21) $TRIG_AFTER $EFFECT_FADE 0.28 0.06
            Animate-Group $slide @(22, 23, 24, 25, 26) $TRIG_AFTER $EFFECT_FADE 0.28 0.06
        }
        
        12 {
            # SLIDE 12: Project Summary
            # Row 1: 7..10, Row 2: 11..14, Row 3: 15..18, Row 4: 19..22, Row 5: 23..26
            # Bottom Pillars: 27
            Animate-Group $slide @(7, 8, 9, 10) $TRIG_AFTER $EFFECT_FADE 0.25 0.1
            Animate-Group $slide @(11, 12, 13, 14) $TRIG_AFTER $EFFECT_FADE 0.25 0.05
            Animate-Group $slide @(15, 16, 17, 18) $TRIG_AFTER $EFFECT_FADE 0.25 0.05
            Animate-Group $slide @(19, 20, 21, 22) $TRIG_AFTER $EFFECT_FADE 0.25 0.05
            Animate-Group $slide @(23, 24, 25, 26) $TRIG_AFTER $EFFECT_FADE 0.25 0.05
            
            # Bottom engineering pillars
            if (27 -le $slide.Shapes.Count) {
                $effPillars = $slide.TimeLine.MainSequence.AddEffect($slide.Shapes[27], $EFFECT_FADE, 0, $TRIG_AFTER)
                $effPillars.Timing.Duration = 0.35
                $effPillars.Timing.TriggerDelayTime = 0.1
            }
        }
    }
}

$pres.Save()
$pres.Close()
$ppt.Quit()

Write-Host "Presentation animations applied and saved successfully!"

