DEFAULT_DATA = {
# Buttons
'AddMarkerButtonDisplay':           'icon',
'AddMarkerButtonIconHover':            './rope/media/add_marker_hover.png',    
'AddMarkerButtonIconOff':              './rope/media/add_marker_off.png',
'AddMarkerButtonIconOn':               './rope/media/add_marker_off.png',
'AddMarkerButtonInfoText':             'ADD MARKER:\nAttaches a parameter marker to the current frame. Markers copy all parameter settings and apply them to all future frames, or until another marker is encountered.',
'AddMarkerButtonState':           False,

'SaveMarkerButtonDisplay':           'icon',
'SaveMarkerButtonIconHover':            './rope/media/marker_save.png',    
'SaveMarkerButtonIconOff':              './rope/media/marker_save.png',
'SaveMarkerButtonIconOn':               './rope/media/marker_save.png',
'SaveMarkerButtonInfoText':             'SAVE MARKERS:\nSave markers for this source video. The markers will be saved as a json file in the same folder as your source video.',
'SaveMarkerButtonState':           False,

'AudioDisplay':             'text', 
'AudioInfoText':             'ENABLE REAL-TIME AUDIO:\nAdds audio from the input video during preview playback. If you are unable to maintain the input video frame rate, the audio will lag.',   
'AudioState':               False,
'AudioText':                'Enable Audio',      
'AutoSwapState':            False,
'ClearFacesDisplay':        'text', 
'ClearFacesIcon':            './rope/media/tarfacedel.png',
'ClearFacesIconHover':      './rope/media/rec.png',
'ClearFacesIconOff':        './rope/media/rec.png',
'ClearFacesIconOn':         './rope/media/rec.png',
'ClearFacesInfoText':             'REMOVE FACES:\nRemove all currently found faces.',
'ClearFacesState':          False,      
'ClearFacesText':           'Clear Faces',   
'ClearmemState':            False,
'DefaultParamsButtonDisplay':           'text',
'DefaultParamsButtonInfoText':             'LOAD DEFAULT PARAMETERS:\nLoad the Rope default parameters for this column.',
'DefaultParamsButtonState':           False,
'DefaultParamsButtonText':           'Load Defaults',
'DelEmbedDisplay':          'text', 
'DelEmbedIconHover':        './rope/media/rec.png',      
'DelEmbedIconOff':          './rope/media/rec.png',
'DelEmbedIconOn':           './rope/media/rec.png',
'DelEmbedInfoText':             'DELETE EMBEDDING:\nDelete the currently selected embedding',
'DelEmbedState':            False,          
'DelEmbedText':             'Delete Emb',  
'DelMarkerButtonDisplay':           'icon',
'DelMarkerButtonIconHover':            './rope/media/remove_marker_hover.png',    
'DelMarkerButtonIconOff':              './rope/media/remove_marker_off.png',
'DelMarkerButtonIconOn':               './rope/media/remove_marker_off.png',
'DelMarkerButtonInfoText':             'REMOVE MARKER:\nRemoves the parameter marker from the current frame.',
'DelMarkerButtonState':           False,
'FindFacesDisplay':         'text', 
'FindFacesIcon':         './rope/media/tarface.png',
'FindFacesIconHover':       './rope/media/rec.png',
'FindFacesIconOff':         './rope/media/rec.png',
'FindFacesIconOn':          './rope/media/rec.png',
'FindFacesInfoText':             'FIND FACES:\nFinds all new faces in the current frame.',
'FindFacesState':           False,  
'FindFacesText':            'Find Faces',  
'ImgDockState':             False,
'ImgVidMode':               'Videos', 
'ImgVidState':              False,
'LoadParamsButtonDisplay':           'text',
'LoadParamsButtonInfoText':             'LOAD SAVED PARAMETERS:\nLoads all parameters from this column if they have been previously saved. ',
'LoadParamsButtonState':           False,
'LoadParamsButtonText':           'Load Params',
'LoadSFacesDisplay':         'both', 
'LoadSFacesIcon':            './rope/media/save.png',
'LoadSFacesIconHover':        './rope/media/save.png',     
'LoadSFacesIconOff':          './rope/media/save.png',
'LoadSFacesIconOn':           './rope/media/save.png',
'LoadSFacesInfoText':             'SELECT SOURCE FACES FOLDER:\nSelects and loads Source Faces from Folder. Make sure the folder only contains <good> images.',
'LoadSFacesState':          False,
'LoadSFacesText':           'Select Faces Folder',  
'LoadTVideosDisplay':         'both', 
'LoadTVideosIconHover':        './rope/media/save.png', 
'LoadTVideosIconOff':          './rope/media/save.png',
'LoadTVideosIconOn':           './rope/media/save.png',
'LoadTVideosInfoText':             'SELECT INPUT VIDEOS/IMAGES FOLDER:\nSelect and load media from folder.',
'LoadTVideosState':         False,    
'LoadTVideosText':           'Select Videos Folder',  
'MaskViewDisplay':         'text',   
'MaskViewInfoText':             'SHOW MASKS:\nDisplays the mask for a face side-by-side with the face. Useful for understanding the masking behaviors and results.',
'MaskViewState':            False,    
'MaskViewText':           'Show Mask', 
'NextMarkerButtonDisplay':           'icon',
'NextMarkerButtonIconHover':            './rope/media/next_marker_hover.png',    
'NextMarkerButtonIconOff':              './rope/media/next_marker_off.png',
'NextMarkerButtonIconOn':               './rope/media/next_marker_off.png',
'NextMarkerButtonInfoText':             'NEXT MARKER:\nMove to the next marker.',
'NextMarkerButtonState':           False,
'OutputFolderDisplay':         'both', 
'OutputFolderIconHover':        './rope/media/save.png', 
'OutputFolderIconOff':          './rope/media/save.png',
'OutputFolderIconOn':           './rope/media/save.png',
'OutputFolderInfoText':             'SELECT SAVE FOLDER:\nSelect folder for saved videos and images.',
'OutputFolderState':        False,   
'OutputFolderText':           'Select Output Folder',  
'PerfTestState':            False,
'PlayDisplay':              'icon',   
'PlayIconHover':            './rope/media/play_hover.png',
'PlayIconOff':              './rope/media/play_off.png',
'PlayIconOn':               './rope/media/play_on.png',
'PlayInfoText':             'PLAY:\nPlays the video. Press again to stop playing',
'PlayState':                False,
'PrevMarkerButtonDisplay':           'icon',
'PrevMarkerButtonIconHover':            './rope/media/previous_marker_hover.png',    
'PrevMarkerButtonIconOff':              './rope/media/previous_marker_off.png',
'PrevMarkerButtonIconOn':               './rope/media/previous_marker_off.png',
'PrevMarkerButtonInfoText':             'PREVIOUS MARKER:\nMove to the previous marker.',
'PrevMarkerButtonState':           False,
'RecordDisplay':            'icon',     
'RecordIconHover':          './rope/media/rec_hover.png',
'RecordIconOff':            './rope/media/rec_off.png',
'RecordIconOn':             './rope/media/rec_on.png',
'RecordInfoText':             'RECORD:\nArms the PLAY button for recording. Press RECORD, then PLAY to record. Press PLAY again to stop recording.',  
'RecordState':              False,       
'SaveImageState':           False,
'SaveParamsButtonDisplay':           'text',
'SaveParamsButtonInfoText':             'SAVE PARAMETERS:\nSaves all parameters in this column.',
'SaveParamsButtonState':            False,
'SaveParamsButtonText':             'Save Params',
'StartRopeDisplay':                 'both', 
'StartRopeIconHover':               './rope/media/rope.png',    
'StartRopeIconOff':                 './rope/media/rope.png',
'StartRopeIconOn':                  './rope/media/rope.png',
'StartRopeInfoText':                'STARTS ROPE:\nStarts up the Rope application.',
'StartRopeState':                   False,    
'StartRopeText':                    'Start Rope',  
'SwapFacesDisplay':                 'text', 
'SwapFacesInfoText':                'SWAP:\nSwap assigned Source Faces and Target Faces.',
'SwapFacesState':                   False,          
'SwapFacesText':                    'Swap Faces',  
'TLBeginningDisplay':              'icon', 
'TLBeginningIconHover':            './rope/media/tl_beg_hover.png',
'TLBeginningIconOff':              './rope/media/tl_beg_off.png',
'TLBeginningIconOn':               './rope/media/tl_beg_on.png',
'TLBeginningInfoText':             'TIMELINE START:\nMove the timeline handle to the first frame.',
'TLBeginningState':                 False,
'TLLeftDisplay':                    'icon',   
'TLLeftIconHover':                  './rope/media/tl_left_hover.png',
'TLLeftIconOff':                    './rope/media/tl_left_off.png',
'TLLeftIconOn':                     './rope/media/tl_left_on.png',
'TLLeftInfoText':                   'TIMELEFT NUDGE LEFT:\nMove the timeline handle to the left 30 frames.',
'TLLeftState':                      False,
'TLRightDisplay':                   'icon',   
'TLRightIconHover':                 './rope/media/tl_right_hover.png',    
'TLRightIconOff':                   './rope/media/tl_right_off.png',
'TLRightIconOn':                    './rope/media/tl_right_on.png',
'TLRightInfoText':                  'TIMELEFT NUDGE RIGHT:\nMove the timeline handle to the RIGHT 30 frames.',
'TLRightState':                     False,

'SaveImageButtonDisplay':                   'text',   
'SaveImageButtonInfoText':                  'SAVE IMAGE:\nSaves the current image to your Output Folder.',
'SaveImageButtonState':                     False,
'SaveImageButtonText':             'Save Image',

'AutoSwapButtonDisplay':                   'text',   
'AutoSwapButtonInfoText':                  'AUTOSWAP:\nAutomatcially applies your currently selected Input Face to new images.',
'AutoSwapButtonState':                     False,
'AutoSwapButtonText':             'Auto Swap',
 
'ClearVramButtonDisplay':                   'text',   
'ClearVramButtonInfoText':                  'CLEAR VRAM:\nClears models from your VRAM.',
'ClearVramButtonState':                     False,
'ClearVramButtonText':             'Clear VRAM',

'GetNewEmbButtonDisplay':                   'text',
'GetNewEmbButtonInfoText':                  'CLEAR VRAM:\nClears models from your VRAM.',
'GetNewEmbButtonState':                     False,
'GetNewEmbButtonText':             'Clear VRAM',


'StopMarkerButtonnDisplay':                   'icon',
'StopMarkerButtonIconHover':            './rope/media/previous_marker_hover.png',
'StopMarkerButtonIconOff':              './rope/media/previous_marker_off.png',
'StopMarkerButtonIconOn':               './rope/media/previous_marker_off.png',
'StopMarkerButtonInfoText':                  'CLEAR VRAM:\nClears models from your VRAM.',
'StopMarkerButtonState':                     False,
'StopMarkerButtonText':             'Clear VRAM',
 
#Switches       
'ColorSwitchInfoText':              'RGB ADJUSTMENT:\nFine-tune the RGB color values of the swap.',
'ColorSwitchState':                 False,
'DiffSwitchInfoText':               'DIFFERENCER:\nAllow some of the original face to show in the swapped result when the difference between the two images is small. Can help bring back some texture to the swapped face',
'DiffSwitchState':                  False,
'FaceAdjSwitchInfoText':            'KPS and SCALE ADJUSTMENT:\nThis is an experimental feature to perform direct adjustments to the face landmarks found by the detector. There is also an option to adjust the scale of the swapped face.',
'FaceAdjSwitchState':               False,
# Face Landmarks Detection
'LandmarksDetectionAdjSwitchInfoText': 'KPS ADJUSTMENT:\nThis is an experimental feature to perform direct adjustments to the face landmarks found by the detector. ',
'LandmarksDetectionAdjSwitchState':    False,
'LandmarksAlignModeFromPointsSwitchInfoText': 'KPS ADJUSTMENT ALIGN MODE FROM POINTS:\nThis is an experimental feature to perform direct adjustments to the face landmarks found from detector key points.',
'LandmarksAlignModeFromPointsSwitchState':    False,
'ShowLandmarksSwitchInfoText':      'Show Landmarks in realtime.',
'ShowLandmarksSwitchState':         False,
#
# Face Landmarks Position
'LandmarksPositionAdjSwitchInfoText': 'KPS ADJUSTMENT:\nThis is an experimental feature to perform direct adjustments to the position of face landmarks found by the detector. ',
'LandmarksPositionAdjSwitchState':    False,
#
# Face Likeness
'FaceLikenessSwitchInfoText':       'Face Likeness:\nThis is an experimental feature to perform direct adjustments to likeness of faces.',
'FaceLikenessSwitchState':           False,
#
# Auto Rotation
'AutoRotationSwitchInfoText':       'Auto Rotation:\nAutomatically Rotate the frames to find the best detection angle',
'AutoRotationSwitchState':           False,
#
'FaceParserSwitchInfoText':         'BACKGROUND MASK:\nAllow the unprocessed background from the orginal image to show in the final swap.',
'FaceParserSwitchState':            False,
'MouthParserSwitchInfoText':        'MOUTH MASK:\nAllow the mouth from the original face to show on the swapped face.',
'MouthParserSwitchState':           False,
'OccluderSwitchInfoText':           'OCCLUSION MASK:\nAllow objects occluding the face to show up in the swapped image.',
'OccluderSwitchState':              False,
'DFLXSegSwitchInfoText':            'DFL XSEG MASK:\nAllow objects occluding the face to show up in the swapped image.',
'DFLXSegSwitchState':               False,
'OrientSwitchInfoText':             'ORIENTATION:\nRotate the face detector to better detect faces at different angles',
'OrientSwitchState':                False,
'RestorerSwitchInfoText':           'FACE RESTORER:\nRestore the swapped image by upscaling.',
'RestorerSwitchState':              False,
'StrengthSwitchInfoText':           'SWAPPER STRENGTH:\nApply additional swapping iterations to increase the strength of the result, which may increase likeness',
'StrengthSwitchState':              False,
'CLIPSwitchInfoText':               'TEXT MASKING:\nUse descriptions to identify objects that will be present in the final swapped image.',
'CLIPSwitchState':                  False,

'VirtualCameraSwitchState':         False,
'VirtualCameraSwitchInfoText':      'VIRTUAL CAMERA:\nFeed the swapped video output to virtual camera for using in external applications',

# Sliders
'BlendSliderAmount':                5,
'BlendSliderInc':                   1,  
'BlendSliderInfoText':              'BLEND:\nCombined masks blending distance. Is not applied to the border masks.',
'BlendSliderMax':                   100,
'BlendSliderMin':                   0,
'BorderBlurSliderAmount':           10,
'BorderBlurSliderInc':              1,  
'BorderBlurSliderInfoText':         'BORDER MASK BLEND:\nBorder mask blending distance.',
'BorderBlurSliderMax':              64,
'BorderBlurSliderMin':              0,
'BorderBottomSliderAmount':         10, 
'BorderBottomSliderInc':            1,  
'BorderBottomSliderInfoText':       'BOTTOM BORDER DISTANCE:\nA rectangle with adjustable top, bottom, and sides that blends the swapped face rseult back into the original image.',
'BorderBottomSliderMax':            64,
'BorderBottomSliderMin':            0,
'BorderSidesSliderAmount':          10, 
'BorderSidesSliderInc':             1,  
'BorderSidesSliderInfoText':        'SIDES BORDER DISTANCE:\nA rectangle with adjustable top, bottom, and sides that blends the swapped face result back into the original image.',
'BorderSidesSliderMax':             64,
'BorderSidesSliderMin':             0,
'BorderTopSliderAmount':            10, 
'BorderTopSliderInc':               1, 
'BorderTopSliderInfoText':          'TOP BORDER DISTANCE:\nA rectangle with adjustable top, bottom, and sides that blends the swapped face result back into the original image.',
'BorderTopSliderMax':               64,
'BorderTopSliderMin':               0,
'ColorBlueSliderAmount':            0,
'ColorBlueSliderInc':               1,  
'ColorBlueSliderInfoText':          'RGB BLUE ADJUSTMENT',
'ColorBlueSliderMax':               100,
'ColorBlueSliderMin':               -100,
'ColorGreenSliderAmount':           0,
'ColorGreenSliderInc':              1, 
'ColorGreenSliderInfoText':         'RGB GREEN ADJUSTMENT',
'ColorGreenSliderMax':              100,
'ColorGreenSliderMin':              -100,
'ColorRedSliderAmount':             0,
'ColorRedSliderInc':                1,  
'ColorRedSliderInfoText':           'RGB RED ADJUSTMENT',
'ColorRedSliderMax':                100,
'ColorRedSliderMin':                -100,
'DetectScoreSliderAmount':          50,
'DetectScoreSliderInc':             1,      
'DetectScoreSliderInfoText':        'DETECTION SCORE LIMIT:\nDetermines the minimum score required for a face to be detected. Higher values require higher quality faces. E.g., if faces are flickering when at extreme angles, raising this will limit swapping attempts.',
'DetectScoreSliderMax':             100,
'DetectScoreSliderMin':             1,
# Face Landmarks Detection
'LandmarksDetectScoreSliderAmount':  50,
'LandmarksDetectScoreSliderInc':     1,      
'LandmarksDetectScoreSliderInfoText':'LANDMARKS DETECTION SCORE LIMIT:\nDetermines the minimum score required for a face to be detected. Higher values require higher quality faces. E.g., if faces are flickering when at extreme angles, raising this will limit swapping attempts.',
'LandmarksDetectScoreSliderMax':     100,
'LandmarksDetectScoreSliderMin':     1,
#
# Face Likeness
'FaceLikenessFactorSliderAmount':    0.00,
'FaceLikenessFactorSliderInc':       0.05,      
'FaceLikenessFactorSliderInfoText':  'Face Likeness Factor:\nDetermines the factor of likeness between the source and assigned faces.',
'FaceLikenessFactorSliderMax':       1.00,
'FaceLikenessFactorSliderMin':       -1.00,
#
# Face Landmarks Position
'FaceIDSliderAmount':               1,
'FaceIDSliderInc':                  1,      
'FaceIDSliderInfoText':             'LANDMARKS POSITION FACE ID:\nDetermines the target face for which the positions of the facial points can be modified.',
'FaceIDSliderMax':                  20,
'FaceIDSliderMin':                  1,
'EyeLeftXSliderAmount':             0, 
'EyeLeftXSliderInc':                1,      
'EyeLeftXSliderInfoText':           'Eye Left X-DIRECTION AMOUNT:\nShifts the eye left detection point left and right',
'EyeLeftXSliderMax':                100,
'EyeLeftXSliderMin':                -100,
'EyeLeftYSliderAmount':             0, 
'EyeLeftYSliderInc':                1,      
'EyeLeftYSliderInfoText':           'Eye Left Y-DIRECTION AMOUNT:\nShifts the eye left detection point up and down',
'EyeLeftYSliderMax':                100,
'EyeLeftYSliderMin':                -100,
'EyeRightXSliderAmount':             0, 
'EyeRightXSliderInc':                1,      
'EyeRightXSliderInfoText':           'Eye Left X-DIRECTION AMOUNT:\nShifts the eye right detection point left and right',
'EyeRightXSliderMax':                100,
'EyeRightXSliderMin':                -100,
'EyeRightYSliderAmount':             0, 
'EyeRightYSliderInc':                1,      
'EyeRightYSliderInfoText':           'Eye Left Y-DIRECTION AMOUNT:\nShifts the eye right detection point up and down',
'EyeRightYSliderMax':                100,
'EyeRightYSliderMin':                -100,
'NoseXSliderAmount':                 0, 
'NoseXSliderInc':                    1,      
'NoseXSliderInfoText':               'Nose X-DIRECTION AMOUNT:\nShifts the nose detection point left and right',
'NoseXSliderMax':                    100,
'NoseXSliderMin':                    -100,
'NoseYSliderAmount':                 0, 
'NoseYSliderInc':                    1,      
'NoseYSliderInfoText':               'Nose Y-DIRECTION AMOUNT:\nShifts the nose detection point up and down',
'NoseYSliderMax':                    100,
'NoseYSliderMin':                    -100,
'MouthLeftXSliderAmount':            0, 
'MouthLeftXSliderInc':               1,      
'MouthLeftXSliderInfoText':          'Mouth Left X-DIRECTION AMOUNT:\nShifts the mouth left detection point left and right',
'MouthLeftXSliderMax':               100,
'MouthLeftXSliderMin':               -100,
'MouthLeftYSliderAmount':            0, 
'MouthLeftYSliderInc':               1,      
'MouthLeftYSliderInfoText':          'Mouth Left Y-DIRECTION AMOUNT:\nShifts the mouth left detection point up and down',
'MouthLeftYSliderMax':               100,
'MouthLeftYSliderMin':               -100,
'MouthRightXSliderAmount':           0, 
'MouthRightXSliderInc':              1,      
'MouthRightXSliderInfoText':         'Mouth Right X-DIRECTION AMOUNT:\nShifts the mouth Right detection point left and right',
'MouthRightXSliderMax':              100,
'MouthRightXSliderMin':              -100,
'MouthRightYSliderAmount':           0, 
'MouthRightYSliderInc':              1,      
'MouthRightYSliderInfoText':         'Mouth Right Y-DIRECTION AMOUNT:\nShifts the mouth Right detection point up and down',
'MouthRightYSliderMax':              100,
'MouthRightYSliderMin':              -100,
#
'DiffSliderAmount':                 4,   
'DiffSliderInc':                    1,
'DiffSliderInfoText':               'DIFFERENCING AMOUNT:\nHigher values relaxes the similarity constraint.',
'DiffSliderMax':                    100,
'DiffSliderMin':                    0,
'FaceParserSliderAmount':           0,   
'FaceParserSliderInc':              1,        
'FaceParserSliderInfoText':         'BACKGROUND MASK AMOUNT:\nNegative/Positive values shrink and grow the mask.',
'FaceParserSliderMax':              50,
'FaceParserSliderMin':              -50,
'FaceScaleSliderAmount':            0,
'FaceScaleSliderInc':               1,    
'FaceScaleSliderInfoText':          'FACE SCALE AMOUNT',
'FaceScaleSliderMax':               20,
'FaceScaleSliderMin':               -20,
'KPSScaleSliderAmount':             0, 
'KPSScaleSliderInc':                1,  
'KPSScaleSliderInfoText':           'KPS SCALE AMOUNT:\nGrows and shrinks the detection point distances.',
'KPSScaleSliderMax':                100,
'KPSScaleSliderMin':                -100,
'KPSXSliderAmount':                 0, 
'KPSXSliderInc':                    1,      
'KPSXSliderInfoText':               'KPS X-DIRECTION AMOUNT:\nShifts the detection points left and right',
'KPSXSliderMax':                    100,
'KPSXSliderMin':                    -100,
'KPSYSliderAmount':                 0, 
'KPSYSliderInc':                    1,  
'KPSYSliderInfoText':               'KPS Y-DIRECTION AMOUNT:\nShifts the detection points lup and down',
'KPSYSliderMax':                    100,
'KPSYSliderMin':                    -100,
'MouthParserSliderAmount':          0, 
'MouthParserSliderInc':             1,      
'MouthParserSliderInfoText':        'MOUTH MASK AMOUNT:\nAdjust the size of the mask. Mask the inside of the mouth, including the tongue',
'MouthParserSliderMax':             30,
'MouthParserSliderMin':             0,

'NeckParserSliderAmount':          0, 
'NeckParserSliderInc':             1,      
'NeckParserSliderInfoText':        'NECK MASK AMOUNT:\nAdjust the size of the mask.',
'NeckParserSliderMax':             30,
'NeckParserSliderMin':             0,

'LeftEyeBrowParserSliderAmount':          0, 
'LeftEyeBrowParserSliderInc':             1,      
'LeftEyeBrowParserSliderInfoText':        'LEFT EYEBROW MASK AMOUNT:\nAdjust the size of the mask.',
'LeftEyeBrowParserSliderMax':             30,
'LeftEyeBrowParserSliderMin':             0,

'RightEyeBrowParserSliderAmount':          0, 
'RightEyeBrowParserSliderInc':             1,      
'RightEyeBrowParserSliderInfoText':        'RIGHT EYEBROW MASK AMOUNT:\nAdjust the size of the mask.',
'RightEyeBrowParserSliderMax':             30,
'RightEyeBrowParserSliderMin':             0,

'LeftEyeParserSliderAmount':          0, 
'LeftEyeParserSliderInc':             1,      
'LeftEyeParserSliderInfoText':        'LEFT EYE MASK AMOUNT:\nAdjust the size of the mask.',
'LeftEyeParserSliderMax':             30,
'LeftEyeParserSliderMin':             0,

'RightEyeParserSliderAmount':          0, 
'RightEyeParserSliderInc':             1,      
'RightEyeParserSliderInfoText':        'RIGHT EYE MASK AMOUNT:\nAdjust the size of the mask.',
'RightEyeParserSliderMax':             30,
'RightEyeParserSliderMin':             0,

'NoseParserSliderAmount':          0, 
'NoseParserSliderInc':             1,      
'NoseParserSliderInfoText':        'NOSE MASK AMOUNT:\nAdjust the size of the mask.',
'NoseParserSliderMax':             30,
'NoseParserSliderMin':             0,

'UpperLipParserSliderAmount':          0, 
'UpperLipParserSliderInc':             1,      
'UpperLipParserSliderInfoText':        'UPPER LIP MASK AMOUNT:\nAdjust the size of the mask.',
'UpperLipParserSliderMax':             30,
'UpperLipParserSliderMin':             0,

'LowerLipParserSliderAmount':          0, 
'LowerLipParserSliderInc':             1,      
'LowerLipParserSliderInfoText':        'LOWER LIP MASK AMOUNT:\nAdjust the size of the mask.',
'LowerLipParserSliderMax':             30,
'LowerLipParserSliderMin':             0,

'OccluderSliderAmount':             0,
'OccluderSliderInc':                1,
'OccluderSliderInfoText':           'OCCLUDER AMOUNT:\nGrows or shrinks the occluded region',
'OccluderSliderMax':                100,
'OccluderSliderMin':                -100,
'OrientSliderAmount':               0,
'OrientSliderInc':                  90,
'OrientSliderInfoText':             'ORIENTATION ANGLE:\nSet this to the angle of the input face angle to help with laying down/upside down/etc. Angles are read clockwise.',
'OrientSliderMax':                  270,
'OrientSliderMin':                  0,
'RestorerSliderAmount':             100,
'RestorerSliderInc':                5,
'RestorerSliderInfoText':           'RESTORER AMOUNT:\nBlends the Restored results back into the original swap.',
'RestorerSliderMax':                100,
'RestorerSliderMin':                0,
'StrengthSliderAmount':             100,
'StrengthSliderInc':                25,    
'StrengthSliderInfoText':           'STRENGTH AMOUNT:\nIncrease up to 5x additional swaps (500%). 200% is generally a good result. Set to 0 to turn off swapping but allow the rest of the pipeline to apply to the original image.',
'StrengthSliderMax':                500,
'StrengthSliderMin':                0,
'ThreadsSliderAmount':              5,
'ThreadsSliderInc':                 1,    
'ThreadsSliderInfoText':            'EXECUTION THREADS:\nSet number of execution threads while playing and recording. Depends strongly on GPU VRAM. 5 threads for 24GB.',
'ThreadsSliderMax':                 50,
'ThreadsSliderMin':                 1,
'ThresholdSliderAmount':            55,
'ThresholdSliderInc':               1,
'ThresholdSliderInfoText':          'THRESHHOLD AMOUNT:\nRaise to reduce faces hopping around when swapping multiple people. A higher value is stricter.',
'ThresholdSliderMax':               100,
'ThresholdSliderMin':               0,
'VideoQualSliderAmount':            18,
'VideoQualSliderInc':               1,      
'VideoQualSliderInfoText':          'VIDEO QUALITY:\nThe encoding quality of the recorded video. 0 is best, 50 is worst, 18 is mostly lossless. File size increases with a lower quality number.',
'VideoQualSliderMax':               50,
'VideoQualSliderMin':               0,

'AudioSpeedSliderAmount':           1.00,
'AudioSpeedSliderInc':              0.01,
'AudioSpeedSliderInfoText':         'AUDIO PLAYBACK SPEED:\nAudo playback when "Enable Audio" is on',
'AudioSpeedSliderMax':              2.00,
'AudioSpeedSliderMin':              0.50,

'CLIPSliderAmount':                 50,
'CLIPSliderInc':                    1,
'CLIPSliderInfoText':               'TEXT MASKING STENGTH:\nIncrease to strengthen the effect.',
'CLIPSliderMax':                    100,
'CLIPSliderMin':                    0,

'ColorGammaSliderAmount':                 1,
'ColorGammaSliderInc':                    0.02,
'ColorGammaSliderInfoText':               'GAMMA VALUE:\nChanges Gamma.',
'ColorGammaSliderMax':                    2,
'ColorGammaSliderMin':                    0,

'ColorBrightSliderAmount':                 1,
'ColorBrightSliderInc':                    0.01,
'ColorBrightSliderInfoText':               'Bright VALUE:\nChanges Bright.',
'ColorBrightSliderMax':                    2,
'ColorBrightSliderMin':                    0,

'ColorContrastSliderAmount':                 1,
'ColorContrastSliderInc':                    0.01,
'ColorContrastSliderInfoText':               'Contrast VALUE:\nChanges Contrast.',
'ColorContrastSliderMax':                    2,
'ColorContrastSliderMin':                    0,

'ColorSaturationSliderAmount':                 1,
'ColorSaturationSliderInc':                    0.01,
'ColorSaturationSliderInfoText':               'Saturation VALUE:\nChanges Saturation.',
'ColorSaturationSliderMax':                    2,
'ColorSaturationSliderMin':                    0,

'ColorSharpnessSliderAmount':                 1,
'ColorSharpnessSliderInc':                    0.1,
'ColorSharpnessSliderInfoText':               'Sharpness VALUE:\nChanges Sharpness.',
'ColorSharpnessSliderMax':                    2,
'ColorSharpnessSliderMin':                    0,

'ColorHueSliderAmount':                 0,
'ColorHueSliderInc':                    0.01,
'ColorHueSliderInfoText':               'Hue VALUE:\nChanges Hue.',
'ColorHueSliderMax':                    0.5,
'ColorHueSliderMin':                    -0.5,

# Text Selection
'DetectTypeTextSelInfoText':        'FACE DETECTION MODEL:\nSelect the face detection model. Mostly only subtle differences, but can significant differences when the face is at extreme angles or covered.',
'DetectTypeTextSelMode':            'Retinaface',
'DetectTypeTextSelModes':           ['Retinaface', 'Yolov8', 'SCRDF', 'Yunet'],
# Face Landmarks Detection
'LandmarksDetectTypeTextSelInfoText': 'LANDMARKS FACE DETECTION MODEL:\nSelect the landmarks face detection model. Mostly only subtle differences, but can significant differences when the face is at extreme angles or covered.',
'LandmarksDetectTypeTextSelMode':     '98',
'LandmarksDetectTypeTextSelModes':    ['5', '68', '3d68', '98', '106', '478'],
#
# Similarity Type
'SimilarityTypeTextSelInfoText':    'Similarity version:\nSelect the similarity to be used with arc face recognizer model.',
'SimilarityTypeTextSelMode':        'Opal',
'SimilarityTypeTextSelModes':       ['Opal', 'Pearl', 'Optimal'],
#
# ProvidersPriority
'ProvidersPriorityTextSelInfoText':    'Providers Priority:\nSelect the providers priority to be used with the system.',
'ProvidersPriorityTextSelMode':        'CUDA',
'ProvidersPriorityTextSelModes':       ['CUDA', 'TensorRT', 'CPU'],
#
# Face Swapper Model
'FaceSwapperModelTextSelInfoText':  'Face Swapper Model:\nSelect the Face Swapper model.',
'FaceSwapperModelTextSelMode':      'Inswapper128',
'FaceSwapperModelTextSelModes':     ['Inswapper128', 'SimSwap512', 'GF1', 'GF2', 'GF3'],
#
'PreviewModeTextSelInfoText':       '',
'PreviewModeTextSelMode':           'Video',
'PreviewModeTextSelModes':          ['Video', 'Image','Theater'],
'RecordTypeTextSelInfoText':        'VIDEO RECORDING LIBRARY:\nSelect the recording library used for video recording. FFMPEG uses the Video Quality slider to adjust the size and quality of the final video. OPENCV has no options but is faster and produces good results.',
'RecordTypeTextSelMode':            'FFMPEG',
'RecordTypeTextSelModes':           ['FFMPEG', 'OPENCV'],
'RestorerDetTypeTextSelInfoText':   'ALIGNMENT:\nSelect how the face is aligned for the Restorer. Original preserves facial features and expressions, but can show some artifacts. Reference softens features. Blend is closer to Reference but is much faster.',
'RestorerDetTypeTextSelMode':       'Blend',
'RestorerDetTypeTextSelModes':      ['Original', 'Blend', 'Reference'],  
'RestorerTypeTextSelInfoText':      'RESTORER TYPE:\nSelect the Restorer type.\nSpeed: GPEN256>GFPGAN>CF>GPEN512>GPEN1024>GPEN2028',
'RestorerTypeTextSelMode':          'GFPGAN',
'RestorerTypeTextSelModes':         ['GFPGAN', 'CF', 'GP256', 'GP512', 'GP1024', 'GP2048'],

'WebCamMaxResolSelInfoText':        "WEBCAM MAX RESOLUTION:\nSelect the maximum resolution to be used by the webcam",
'WebCamMaxResolSelMode':            '1920x1080',
'WebCamMaxResolSelModes':           ['384x216', '640x360', '1280x720', '1920x1080'],
'MergeTextSelInfoText':      'INPUT FACES MERGE MATH:\nWhen shift-clicking face for merging, determines how the embedding vectors are combined.',
'MergeTextSelMode':          'Mean',
'MergeTextSelModes':         ['Mean', 'Median'],
'SwapperTypeTextSelInfoText':      'SWAPPER OUTPUT RESOLUTION:\nDetermines the resolution of the swapper output.',
'SwapperTypeTextSelMode':          '128',
'SwapperTypeTextSelModes':         ['128', '256', '512'],



# Text Entry
'CLIPTextEntry':    '',
'CLIPTextEntryInfoText':            'TEXT MASKING ENTRY:\nTo use, type a word(s) in the box separated by commas and press <enter>.',
}

PARAM_VARS =    {

    'CLIPState':                False,
    'CLIPMode':                 0, 
    'CLIPModes':                ['CLIP'], 
    'CLIPAmount':               [50],
    'CLIPMin':                  0,
    'CLIPMax':                  100,
    'CLIPInc':                  1,                                                  
    'CLIPUnit':                 '%', 
    'CLIPIcon':                 './rope/media/CLIP.png',
    'CLIPMessage':              'CLIP - Text based occluder. Occluded objects are visible in the final image (occluded from the mask). [LB: on/off, MW: strength]',                              
    'CLIPFunction':         False,

    "CLIPText":                 '',
}   
 
PARAMS =   { 
   
    'ClearmemFunction':         'self.clear_mem()',
    'PerfTestFunction':         'self.toggle_perf_test()',
    'ImgVidFunction':         'self.toggle_vid_img()',
    'AutoSwapFunction':         'self.toggle_auto_swap()',
    'SaveImageFunction':         'self.save_image()',

    'ClearmemIcon':            './rope/media/clear_mem.png',
    'SaveImageIcon':            './rope/media/save_disk.png', 
    'PerfTestIcon':            './rope/media/test.png',
    'RefDelIcon':          './rope/media/construction.png',
    'TransformIcon':          './rope/media/scale.png',    
    'ThresholdIcon':            './rope/media/thresh.png',
    'LoadSFacesIcon':            './rope/media/save.png',
    'BorderIcon':                 './rope/media/maskup.png',
    'OccluderIcon':             './rope/media/occluder.png',
    'ColorIcon':            './rope/media/rgb.png',
    'StrengthIcon':             './rope/media/strength.png',
    'OrientationIcon':          './rope/media/orient.png',
    'DiffIcon':                 './rope/media/diff.png',
    'MouthParserIcon':           './rope/media/parse.png',
    'AudioIcon':            './rope/media/rgb.png',    
    'VideoQualityIcon':            './rope/media/tarface.png',    
    'MaskViewIcon':             './rope/media/maskblur.png',
    'BlurIcon':                 './rope/media/blur.png',    
    'ToggleStopIcon':            './rope/media/STOP.png',    
     'DelEmbedIcon':            './rope/media/delemb.png',   
    'ImgVidIcon':            './rope/media/imgvid.png',    
    
    
    
    'ImgVidMessage':         'IMAGE/VIDEO - Toggle between Image and Video folder view.',      
    'ToggleStopMessage':         'STOP MARKER - Sets a frame that will stop the video playing/recording.',      
    'AutoSwapMessage':         'AUTO SWAP - Automatically swaps the first person in an image to the selcted source faces [LB: Turn on/off]',      
    'SaveImageMessage':         'SAVE IMAGE - Save image to output folder',      
    'ClearmemMessage':         'CLEAR VRAM - Clears all models from VRAM [LB: Clear]',      
    'PerfTestMessage':         'PERFORMANCE DATA - Displays timing data in the console for critical Rope functions. [LB: on/off]',
    'RefDelMessage':       'REFERENCE DELTA - Modify the reference points. Turn on mask preview to see adjustments. [LB: on/off, RB: translate x/y, and scale, MW: amount]' ,
    'ThresholdMessage':         'THRESHOLD - Threshold for determining if Target Faces match faces in a frame. Lower is stricter. [LB: use amount/match all, MW: value]',
    'TransformMessage':       'SCALE - Adjust the scale of the face. Use with Background parser to blend into the image. [LB: on/off, MW: amount]',     
    'PlayMessage':         'PLAY - Plays the video. Press again to stop playing',  
 
     }   
     
