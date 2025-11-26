// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#include <PTexLib.h>

#include <pangolin/display/display.h>
#include <pangolin/display/widgets.h>
#include <pangolin/utils/picojson.h>
#include <Eigen/Dense>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>

#include "GLCheck.h"
#include "MirrorRenderer.h"

int main(int argc, char* argv[]) {

  ASSERT(argc == 3 || argc == 4 || argc == 5 || argc == 6, "Usage: ./ReplicaViewer mesh.ply textures [glass.sur] [output_transforms.json] [--append]");

  const std::string meshFile(argv[1]);
  const std::string atlasFolder(argv[2]);
  ASSERT(pangolin::FileExists(meshFile));
  ASSERT(pangolin::FileExists(atlasFolder));

  std::string surfaceFile;
  std::string outputPath = "transforms.json";
  bool appendMode = false;
  
  if (argc >= 4) {
    surfaceFile = std::string(argv[3]);
    if (surfaceFile != "none" && surfaceFile.length() > 0) {
      ASSERT(pangolin::FileExists(surfaceFile));
    } else {
      surfaceFile = "";
    }
  }
  
  if (argc >= 5) {
    std::string arg4 = std::string(argv[4]);
    if (arg4 == "--append") {
      appendMode = true;
    } else {
      outputPath = arg4;
    }
  }
  
  if (argc == 6) {
    std::string arg5 = std::string(argv[5]);
    if (arg5 == "--append") {
      appendMode = true;
    }
  }

  const int uiWidth = 180;
  const int width = 1280;
  const int height = 960;
  
  // Camera intrinsics
  float fx = width / 2.0f;
  float fy = width / 2.0f;
  float cx = (width - 1.0f) / 2.0f;
  float cy = (height - 1.0f) / 2.0f;
  
  // Storage for saved poses
  std::vector<Eigen::Matrix4d> savedPoses;
  int frameCounter = 0;
  int colmapImId = 1;
  
  // Load existing transforms if in append mode
  if (appendMode && pangolin::FileExists(outputPath)) {
    std::cout << "Append mode: loading existing poses from " << outputPath << std::endl;
    std::ifstream file(outputPath);
    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string json_str = buffer.str();
    
    picojson::value v;
    std::string err = picojson::parse(v, json_str);
    if (err.empty() && v.is<picojson::object>()) {
      const picojson::object& obj = v.get<picojson::object>();
      if (obj.count("frames") > 0) {
        const picojson::array& frames = obj.at("frames").get<picojson::array>();
        for (size_t i = 0; i < frames.size(); i++) {
          const picojson::object& frame = frames[i].get<picojson::object>();
          const picojson::array& matrix = frame.at("transform_matrix").get<picojson::array>();
          
          Eigen::Matrix4d pose;
          for (int row = 0; row < 4; row++) {
            const picojson::array& row_data = matrix[row].get<picojson::array>();
            for (int col = 0; col < 4; col++) {
              pose(row, col) = row_data[col].get<double>();
            }
          }
          savedPoses.push_back(pose);
        }
        std::cout << "Loaded " << savedPoses.size() << " existing poses" << std::endl;
      }
    }
  }

  // Setup OpenGL Display (based on GLUT)
  pangolin::CreateWindowAndBind("ReplicaViewer", uiWidth + width, height);

  // Epoxy is used instead of GLEW, no initialization needed

  if(!checkGLVersion()) {
    return 1;
  }

  // Setup default OpenGL parameters
  glEnable(GL_DEPTH_TEST);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  glPixelStorei(GL_PACK_ALIGNMENT, 1);
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
  const GLenum frontFace = GL_CW;
  glFrontFace(frontFace);
  glLineWidth(1.0f);

  // Tell the base view to arrange its children equally
  if (uiWidth != 0) {
    pangolin::CreatePanel("ui").SetBounds(0, 1.0f, 0, pangolin::Attach::Pix(uiWidth));
  }

  pangolin::View& container =
      pangolin::CreateDisplay().SetBounds(0, 1.0f, pangolin::Attach::Pix(uiWidth), 1.0f);

  pangolin::OpenGlRenderState s_cam(
      pangolin::ProjectionMatrixRDF_TopLeft(
          width,
          height,
          fx,
          fy,
          cx,
          cy,
          0.1f,
          100.0f),
      pangolin::ModelViewLookAtRDF(0, 0, 4, 0, 0, 0, 0, 1, 0));

  pangolin::Handler3D s_handler(s_cam);

  pangolin::View& meshView = pangolin::Display("MeshView")
                                 .SetBounds(0, 1.0f, 0, 1.0f, (double)width / (double)height)
                                 .SetHandler(&s_handler);

  container.AddDisplay(meshView);

  // load mirrors
  std::vector<MirrorSurface> mirrors;
  if (surfaceFile.length()) {
    std::ifstream file(surfaceFile);
    picojson::value json;
    picojson::parse(json, file);

    for (size_t i = 0; i < json.size(); i++) {
      mirrors.emplace_back(json[i]);
    }
    std::cout << "Loaded " << mirrors.size() << " mirrors" << std::endl;
  }

  const std::string shadir = STR(SHADER_DIR);
  MirrorRenderer mirrorRenderer(mirrors, width, height, shadir);

  // load mesh and textures
  PTexMesh ptexMesh(meshFile, atlasFolder);

  pangolin::Var<float> exposure("ui.Exposure", 0.01, 0.0f, 0.1f);
  pangolin::Var<float> gamma("ui.Gamma", ptexMesh.Gamma(), 1.0f, 3.0f);
  pangolin::Var<float> saturation("ui.Saturation", ptexMesh.Saturation(), 0.0f, 2.0f);
  pangolin::Var<float> depthScale("ui.Depth_scale", 0.1f, 0.0f, 1.0f);

  pangolin::Var<bool> wireframe("ui.Wireframe", false, true);
  pangolin::Var<bool> drawBackfaces("ui.Draw_backfaces", false, true);
  pangolin::Var<bool> drawMirrors("ui.Draw_mirrors", true, true);
  pangolin::Var<bool> drawDepth("ui.Draw_depth", false, true);

  ptexMesh.SetExposure(exposure);
  
  // Register keyboard callbacks for saving poses
  bool shouldSavePose = false;
  bool shouldQuit = false;
  
  pangolin::RegisterKeyPressCallback('s', [&shouldSavePose]() { shouldSavePose = true; });
  pangolin::RegisterKeyPressCallback('S', [&shouldSavePose]() { shouldSavePose = true; });
  pangolin::RegisterKeyPressCallback('q', [&shouldQuit]() { shouldQuit = true; });
  pangolin::RegisterKeyPressCallback('Q', [&shouldQuit]() { shouldQuit = true; });
  
  std::cout << "\n" << std::string(60, '=') << std::endl;
  std::cout << "CONTROLS:" << std::endl;
  std::cout << "  Mouse: Rotate/Pan/Zoom the view" << std::endl;
  std::cout << "  S: Save current camera pose" << std::endl;
  std::cout << "  Q: Save all poses to transforms.json and quit" << std::endl;
  std::cout << std::string(60, '=') << std::endl;
  std::cout << "Output will be saved to: " << outputPath << std::endl;
  std::cout << "\nNavigate to desired viewpoints and press 'S' to collect poses...\n" << std::endl;

  while (!pangolin::ShouldQuit() && !shouldQuit) {
    glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
    
    // Check if we should save pose
    if (shouldSavePose) {
      shouldSavePose = false;
      
      // Get current camera modelview matrix (world-to-camera)
      pangolin::OpenGlMatrix mvMat = s_cam.GetModelViewMatrix();
      Eigen::Matrix4d w2c = Eigen::Map<const Eigen::Matrix<double,4,4,Eigen::ColMajor>>(mvMat.m);
      
      // Convert to camera-to-world, then flip Y and Z to match NeRFstudio (OpenGL) convention
      Eigen::Matrix4d c2w = w2c.inverse();
      
      // NeRFstudio expects OpenGL convention (camera looks down -Z)
      // Apply coordinate flip: negate Y and Z columns
      c2w.col(1) *= -1;  // Flip Y
      c2w.col(2) *= -1;  // Flip Z
      
      savedPoses.push_back(c2w);
      std::cout << "✓ Saved pose " << savedPoses.size() 
                << " at position: [" << c2w(0,3) << ", " 
                << c2w(1,3) << ", " << c2w(2,3) << "]" << std::endl;
    }

    if (exposure.GuiChanged()) {
      ptexMesh.SetExposure(exposure);
    }

    if (gamma.GuiChanged()) {
      ptexMesh.SetGamma(gamma);
    }

    if (saturation.GuiChanged()) {
      ptexMesh.SetSaturation(saturation);
    }

    if (meshView.IsShown()) {
      meshView.Activate(s_cam);

      if (drawBackfaces) {
        glDisable(GL_CULL_FACE);
      } else {
        glEnable(GL_CULL_FACE);
      }

      if (wireframe) {
        glEnable(GL_POLYGON_OFFSET_FILL);
        glPolygonOffset(1.0, 1.0);
        ptexMesh.Render(s_cam);
        glDisable(GL_POLYGON_OFFSET_FILL);
        // render wireframe on top
        ptexMesh.RenderWireframe(s_cam);
      } else if (drawDepth) {
        ptexMesh.RenderDepth(s_cam, depthScale);
      } else {
        ptexMesh.Render(s_cam);
      }

      glDisable(GL_CULL_FACE);

      if (drawMirrors) {
        for (size_t i = 0; i < mirrors.size(); i++) {
          MirrorSurface& mirror = mirrors[i];
          // capture reflections
          mirrorRenderer.CaptureReflection(mirror, ptexMesh, s_cam, frontFace, drawDepth, depthScale);

          // render mirror
          mirrorRenderer.Render(mirror, mirrorRenderer.GetMaskTexture(i), s_cam, drawDepth);
        }
      }
    }

    pangolin::FinishFrame();
  }
  
  // Save transforms.json if quitting with Q
  if (shouldQuit && savedPoses.size() > 0) {
    // Create transforms.json
    std::stringstream transforms_json;
    transforms_json << std::fixed << std::setprecision(10);
    transforms_json << "{\n";
    transforms_json << "  \"w\": " << width << ",\n";
    transforms_json << "  \"h\": " << height << ",\n";
    transforms_json << "  \"fl_x\": " << fx << ",\n";
    transforms_json << "  \"fl_y\": " << fy << ",\n";
    transforms_json << "  \"cx\": " << cx << ",\n";
    transforms_json << "  \"cy\": " << cy << ",\n";
    transforms_json << "  \"k1\": 0.0,\n";
    transforms_json << "  \"k2\": 0.0,\n";
    transforms_json << "  \"p1\": 0.0,\n";
    transforms_json << "  \"p2\": 0.0,\n";
    transforms_json << "  \"camera_model\": \"OPENCV\",\n";
    transforms_json << "  \"frames\": [\n";
    
    for (size_t i = 0; i < savedPoses.size(); i++) {
      transforms_json << "    {\n";
      transforms_json << "      \"file_path\": \"images/frame_" 
                     << std::setfill('0') << std::setw(6) << i << ".png\",\n";
      transforms_json << "      \"transform_matrix\": [\n";
      
      for (int row = 0; row < 4; row++) {
        transforms_json << "        [";
        for (int col = 0; col < 4; col++) {
          transforms_json << savedPoses[i](row, col);
          if (col < 3) transforms_json << ", ";
        }
        transforms_json << "]";
        if (row < 3) transforms_json << ",";
        transforms_json << "\n";
      }
      
      transforms_json << "      ],\n";
      transforms_json << "      \"colmap_im_id\": " << (i + 1) << "\n";
      transforms_json << "    }";
      if (i < savedPoses.size() - 1) transforms_json << ",";
      transforms_json << "\n";
    }
    
    transforms_json << "  ]\n";
    transforms_json << "}\n";
    
    // Write to file
    std::ofstream transforms_file(outputPath);
    transforms_file << transforms_json.str();
    transforms_file.close();
    
    std::cout << "\n✅ Saved " << savedPoses.size() 
              << " camera poses to " << outputPath << std::endl;
  } else if (shouldQuit) {
    std::cout << "No poses saved. Exiting without saving." << std::endl;
  }

  return 0;
}
