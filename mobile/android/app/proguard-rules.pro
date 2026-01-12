# Keep all onnxruntime Java bindings
-keep class ai.onnxruntime.** { *; }
-keepclassmembers class ai.onnxruntime.** { *; }

# Prevent JNI reflection failures
-keepattributes *Annotation*
