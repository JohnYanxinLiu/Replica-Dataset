// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#include <EGL.h>
#include <PTexLib.h>
#include <pangolin/image/image_convert.h>
#include <Eigen/Dense>
#include <fstream>
#include <sstream>
#include <iomanip>

#include "GLCheck.h"
#include "MirrorRenderer.h"


int main(int argc, char* argv[]) {
  ASSERT(argc == 3 || argc == 4 || argc == 5 || argc == 6 || argc == 7, "Usage: ./ReplicaRenderer mesh.ply /path/to/atlases [mirrorFile] [transforms.json] [output_dir] [--continue]");

  const std::string meshFile(argv[1]);
  const std::string atlasFolder(argv[2]);
  ASSERT(pangolin::FileExists(meshFile));
  ASSERT(pangolin::FileExists(atlasFolder));

  std::string surfaceFile;
  std::string trajectoryFile;
  std::string outputDir = ".";
  bool continueMode = false;
  
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
    if (arg4 == "--continue") {
      continueMode = true;
    } else {
      trajectoryFile = arg4;
      if (trajectoryFile.length() > 0) {
        ASSERT(pangolin::FileExists(trajectoryFile));
      }
    }
  }
  
  if (argc >= 6) {
    std::string arg5 = std::string(argv[5]);
    if (arg5 == "--continue") {
      continueMode = true;
    } else {
      outputDir = arg5;
    }
  }
  
  if (argc == 7) {
    std::string arg6 = std::string(argv[6]);
    if (arg6 == "--continue") {
      continueMode = true;
    }
  }
  
  // Create output directory structure
  std::string imagesDir = outputDir + "/images";
  std::system(("mkdir -p " + imagesDir).c_str());

  const int width = 1280;
  const int height = 960;
  bool renderDepth = true;
  float depthScale = 65535.0f * 0.1f;
  
  // Camera intrinsics for transforms.json
  float fx = width / 2.0f;
  float fy = width / 2.0f;
  float cx = (width - 1.0f) / 2.0f;
  float cy = (height - 1.0f) / 2.0f;
  float camera_angle_x = 2.0f * atan(width / (2.0f * fx));

  // Setup EGL
  EGLCtx egl;

  egl.PrintInformation();
  
  if(!checkGLVersion()) {
    return 1;
  }

  //Don't draw backfaces
  const GLenum frontFace = GL_CCW;
  glFrontFace(frontFace);

  // Setup a framebuffer
  pangolin::GlTexture render(width, height);
  pangolin::GlRenderBuffer renderBuffer(width, height);
  pangolin::GlFramebuffer frameBuffer(render, renderBuffer);

  pangolin::GlTexture depthTexture(width, height, GL_R32F, false, 0, GL_RED, GL_FLOAT, 0);
  pangolin::GlFramebuffer depthFrameBuffer(depthTexture, renderBuffer);

  // Setup a camera
  pangolin::OpenGlRenderState s_cam(
      pangolin::ProjectionMatrixRDF_BottomLeft(
          width,
          height,
          fx,
          fy,
          cx,
          cy,
          0.1f,
          100.0f),
      pangolin::ModelViewLookAtRDF(0, 0, 4, 0, 0, 0, 0, 1, 0));

  // Load trajectory from file if provided
  std::vector<Eigen::Matrix4d> camera_poses;
  size_t numFrames = 100;
  
  if (trajectoryFile.length() > 0) {
    std::ifstream file(trajectoryFile);
    picojson::value json;
    std::string err = picojson::parse(json, file);
    if (!err.empty()) {
      std::cerr << "Error parsing trajectory file: " << err << std::endl;
      return 1;
    }
    
    if (json.is<picojson::object>()) {
      const picojson::object& obj = json.get<picojson::object>();
      if (obj.find("frames") != obj.end() && obj.at("frames").is<picojson::array>()) {
        const picojson::array& frames = obj.at("frames").get<picojson::array>();
        numFrames = frames.size();
      
      for (size_t i = 0; i < frames.size(); i++) {
        const picojson::object& frame = frames[i].get<picojson::object>();
        const picojson::array& matrix = frame.at("transform_matrix").get<picojson::array>();
        
        // Load camera-to-world matrix from JSON (NeRFstudio OpenGL convention)
        Eigen::Matrix4d c2w;
        for (int row = 0; row < 4; row++) {
          const picojson::array& row_data = matrix[row].get<picojson::array>();
          for (int col = 0; col < 4; col++) {
            c2w(row, col) = row_data[col].get<double>();
          }
        }
        
        // Undo the coordinate flip to get back to mesh native coordinates
        c2w.col(1) *= -1;  // Flip Y back
        c2w.col(2) *= -1;  // Flip Z back
        
        // Convert to world-to-camera for Pangolin
        Eigen::Matrix4d w2c = c2w.inverse();
        camera_poses.push_back(w2c);
      }
      std::cout << "Loaded " << numFrames << " camera poses from trajectory file" << std::endl;
      }
    }
  }

  // If no trajectory file, generate default linear trajectory
  if (camera_poses.empty()) {
    pangolin::OpenGlMatrix mvMat = s_cam.GetModelViewMatrix();
    Eigen::Matrix4d T_camera_world = Eigen::Map<const Eigen::Matrix<double,4,4,Eigen::ColMajor>>(mvMat.m);
    Eigen::Matrix4d T_new_old = Eigen::Matrix4d::Identity();
    T_new_old.topRightCorner(3, 1) = Eigen::Vector3d(0.025, 0, 0);
    
    for (size_t i = 0; i < numFrames; i++) {
      camera_poses.push_back(T_camera_world);
      T_camera_world = T_camera_world * T_new_old.inverse();
    }
  }

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

  pangolin::ManagedImage<Eigen::Matrix<uint8_t, 3, 1>> image(width, height);
  pangolin::ManagedImage<float> depthImage(width, height);
  pangolin::ManagedImage<uint16_t> depthImageInt(width, height);

  // Prepare transforms.json output
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

  // Render frames
  for (size_t i = 0; i < numFrames; i++) {
    // Check if file already exists in continue mode
    char test_filename[1000];
    snprintf(test_filename, 1000, "%s/images/frame_%06zu.png", outputDir.c_str(), i);
    
    if (continueMode && pangolin::FileExists(test_filename)) {
      std::cout << "\rSkipping frame " << i + 1 << "/" << numFrames << " (already exists)... ";
      std::cout.flush();
      
      // Still need to add to transforms.json, so get the camera pose
      Eigen::Matrix4d w2c = camera_poses[i];
      Eigen::Matrix4d c2w = w2c.inverse();
      
      // NeRFstudio expects OpenGL convention (camera looks down -Z)
      // Apply coordinate flip: negate Y and Z columns
      c2w.col(1) *= -1;  // Flip Y
      c2w.col(2) *= -1;  // Flip Z
      
      transforms_json << "    {\n";
      transforms_json << "      \"file_path\": \"images/frame_" << std::setfill('0') << std::setw(6) << i << ".png\",\n";
      transforms_json << "      \"transform_matrix\": [\n";
      for (int row = 0; row < 4; row++) {
        transforms_json << "        [";
        for (int col = 0; col < 4; col++) {
          transforms_json << c2w(row, col);
          if (col < 3) transforms_json << ", ";
        }
        transforms_json << "]";
        if (row < 3) transforms_json << ",";
        transforms_json << "\n";
      }
      transforms_json << "      ],\n";
      transforms_json << "      \"colmap_im_id\": " << (i + 1) << "\n";
      transforms_json << "    }";
      if (i < numFrames - 1) transforms_json << ",";
      transforms_json << "\n";
      
      continue;
    }
    
    std::cout << "\rRendering frame " << i + 1 << "/" << numFrames << "... ";
    std::cout.flush();

    // Set camera pose for this frame
    pangolin::OpenGlMatrix mvMat;
    Eigen::Map<Eigen::Matrix<double,4,4,Eigen::ColMajor>>(mvMat.m) = camera_poses[i];
    s_cam.GetModelViewMatrix() = mvMat;

    // Render
    frameBuffer.Bind();
    glPushAttrib(GL_VIEWPORT_BIT);
    glViewport(0, 0, width, height);
    glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

    glEnable(GL_CULL_FACE);

    ptexMesh.Render(s_cam);

    glDisable(GL_CULL_FACE);

    glPopAttrib(); //GL_VIEWPORT_BIT
    frameBuffer.Unbind();

    for (size_t i = 0; i < mirrors.size(); i++) {
      MirrorSurface& mirror = mirrors[i];
      // capture reflections
      mirrorRenderer.CaptureReflection(mirror, ptexMesh, s_cam, frontFace);

      frameBuffer.Bind();
      glPushAttrib(GL_VIEWPORT_BIT);
      glViewport(0, 0, width, height);

      // render mirror
      mirrorRenderer.Render(mirror, mirrorRenderer.GetMaskTexture(i), s_cam);

      glPopAttrib(); //GL_VIEWPORT_BIT
      frameBuffer.Unbind();
    }

    // Download and save
    render.Download(image.ptr, GL_RGB, GL_UNSIGNED_BYTE);

    char filename[1000];
    snprintf(filename, 1000, "%s/images/frame_%06zu.png", outputDir.c_str(), i);

    pangolin::SaveImage(
        image.UnsafeReinterpret<uint8_t>(),
        pangolin::PixelFormatFromString("RGB24"),
        std::string(filename));

    if (renderDepth) {
      // render depth
      depthFrameBuffer.Bind();
      glPushAttrib(GL_VIEWPORT_BIT);
      glViewport(0, 0, width, height);
      glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

      glEnable(GL_CULL_FACE);

      ptexMesh.RenderDepth(s_cam, depthScale);

      glDisable(GL_CULL_FACE);

      glPopAttrib(); //GL_VIEWPORT_BIT
      depthFrameBuffer.Unbind();

      depthTexture.Download(depthImage.ptr, GL_RED, GL_FLOAT);

      // convert to 16-bit int
      for(size_t i = 0; i < depthImage.Area(); i++)
          depthImageInt[i] = static_cast<uint16_t>(depthImage[i] + 0.5f);

      snprintf(filename, 1000, "%s/images/depth_%06zu.png", outputDir.c_str(), i);
      pangolin::SaveImage(
          depthImageInt.UnsafeReinterpret<uint8_t>(),
          pangolin::PixelFormatFromString("GRAY16LE"),
          std::string(filename), true, 34.0f);
    }

    // Add frame to transforms.json
    // camera_poses[i] is world-to-camera (Pangolin ModelView)
    Eigen::Matrix4d w2c = camera_poses[i];
    Eigen::Matrix4d c2w = w2c.inverse();
    
    // NeRFstudio expects OpenGL convention (camera looks down -Z)
    // Apply coordinate flip: negate Y and Z columns
    c2w.col(1) *= -1;  // Flip Y
    c2w.col(2) *= -1;  // Flip Z
    
    transforms_json << "    {\n";
    transforms_json << "      \"file_path\": \"images/frame_" << std::setfill('0') << std::setw(6) << i << ".png\",\n";
    transforms_json << "      \"transform_matrix\": [\n";
    for (int row = 0; row < 4; row++) {
      transforms_json << "        [";
      for (int col = 0; col < 4; col++) {
        transforms_json << c2w(row, col);
        if (col < 3) transforms_json << ", ";
      }
      transforms_json << "]";
      if (row < 3) transforms_json << ",";
      transforms_json << "\n";
    }
    transforms_json << "      ],\n";
    transforms_json << "      \"colmap_im_id\": " << (i + 1) << "\n";
    transforms_json << "    }";
    if (i < numFrames - 1) transforms_json << ",";
    transforms_json << "\n";
  }
  
  // Close transforms.json
  transforms_json << "  ]\n";
  transforms_json << "}\n";
  
  // Write transforms.json to file
  std::string transforms_path = outputDir + "/transforms.json";
  std::ofstream transforms_file(transforms_path);
  transforms_file << transforms_json.str();
  transforms_file.close();
  
  std::cout << "\rRendering frame " << numFrames << "/" << numFrames << "... done" << std::endl;
  std::cout << "Saved transforms.json to " << transforms_path << " with " << numFrames << " frames" << std::endl;

  return 0;
}

