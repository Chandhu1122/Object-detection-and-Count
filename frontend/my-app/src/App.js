import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { CheckCircleIcon, PlayIcon, StopIcon, XCircleIcon, InformationCircleIcon, VideoCameraIcon, UserIcon, TruckIcon, RectangleStackIcon } from '@heroicons/react/24/solid';
import backgroundImage from './bg1.webp';

const defaultSupervisors = ['Shiva', 'Ram','Vishnu', 'Other'];
const defaultVehicles = ['TN 36 P 8220', 'KA 19 P 8488', 'KL 07 CP 7235', 'Other'];
const defaultModels = ["4,5,6 box","single box","multiple box"];

const Toast = ({ message, type, onClose }) => {
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    if (message) {
      setIsVisible(true);
      const timer = setTimeout(() => {
        setIsVisible(false);
        onClose();
      }, 3000);
      return () => clearTimeout(timer);
    } else {
      setIsVisible(false);
    }
  }, [message, onClose]);

  if (!isVisible) return null;

  let bgColor, icon;
  switch (type) {
    case 'success':
      bgColor = '#10B981';
      icon = <CheckCircleIcon style={{ height: '20px', width: '20px' }} />;
      break;
    case 'info':
      bgColor = '#3B82F6';
      icon = <InformationCircleIcon style={{ height: '20px', width: '20px' }} />;
      break;
    case 'error':
      bgColor = '#EF4444';
      icon = <XCircleIcon style={{ height: '20px', width: '20px' }} />;
      break;
    default:
      bgColor = '#6B7280';
      icon = <InformationCircleIcon style={{ height: '20px', width: '20px' }} />;
  }

  return (
    <div style={{
      position: 'fixed',
      bottom: '16px',
      right: '16px',
      zIndex: 50,
      padding: '16px',
      borderRadius: '16px',
      boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
      color: 'white',
      display: 'flex',
      alignItems: 'center',
      gap: '12px',
      transition: 'opacity 0.3s',
      backgroundColor: bgColor,
    }}>
      {icon}
      <span>{message}</span>
    </div>
  );
};

export default function App() {
  const [videoLink, setVideoLink] = useState('');
  const [supervisor, setSupervisor] = useState(defaultSupervisors[0]);
  const [customSupervisor, setCustomSupervisor] = useState('');
  const [vehicle, setVehicle] = useState(defaultVehicles[0]);
  const [customVehicle, setCustomVehicle] = useState('');
  const [model, setModel] = useState(defaultModels[0]);
  const [isDetecting, setIsDetecting] = useState(false);
  const [toastMessage, setToastMessage] = useState({ message: '', type: '' });

  const showToast = (message, type) => {
    setToastMessage({ message, type });
  };

  const handleStart = async () => {
    const finalSupervisor = supervisor === 'Other' ? customSupervisor : supervisor;
    const finalVehicle = vehicle === 'Other' ? customVehicle : vehicle;

    if (!videoLink.trim()) {
      showToast('Please enter a video link.', 'error');
      return;
    }
    if (!finalSupervisor.trim()) {
      showToast('Please select or enter a supervisor name.', 'error');
      return;
    }
    if (!finalVehicle.trim()) {
      showToast('Please select or enter a vehicle number.', 'error');
      return;
    }
    if (!model.trim()) {
      showToast('Please select a model.', 'error');
      return;
    }

    setIsDetecting(true);
    showToast('Video detection started successfully!', 'success');

    try {
      const response = await axios.post('https://box-detection-api.onrender.com/start-detection', {
        video_url: videoLink,
        supervisor_name: finalSupervisor,
        vehicle_number: finalVehicle,
        model_name: model,
      });
      showToast(response.data.message, 'success');
    } catch (error) {
      console.error('Error starting detection:', error);
      showToast('Failed to start detection. Check the console for details.', 'error');
      setIsDetecting(false);
    }
  };

  const handleStop = async () => {
    setIsDetecting(false);
    showToast('Video detection stopped.', 'info');

    const finalSupervisor = supervisor === 'Other' ? customSupervisor : supervisor;
    const finalVehicle = vehicle === 'Other' ? customVehicle : vehicle;

    try {
      const response = await axios.post('https://box-detection-api.onrender.com/stop-detection', {
        vehicle_number: finalVehicle,
        supervisor_name: finalSupervisor,
      });
      showToast(response.data.message, 'success');
    } catch (error) {
      console.error('Error stopping detection:', error);
      showToast('Failed to stop detection. Check the console for details.', 'error');
    }
  };

  return (
    <div style={styles.appContainer}>
      <div style={styles.card}>
        <div style={styles.cardHeader}>
          <div style={styles.cardHeaderIconContainer}>
            <VideoCameraIcon style={styles.cardHeaderIcon} />
          </div>
          <h1 style={styles.cardTitle}>
            Video Detection System
          </h1>
          <p style={styles.cardDescription}>
            AI-Powered Video Analysis & Monitoring
          </p>
        </div>

        <div style={styles.cardContent}>
          <div style={styles.gridContainer}>
            {/* Video Link and Model fields are now a 2-column grid */}
            <div style={{ ...styles.formGroup, gridColumn: 'span 2' }}>
              <label style={styles.label}>Video Link</label>
              <div style={styles.inputWrapper}>
                <VideoCameraIcon style={styles.inputIcon} />
                <input
                  type="text"
                  value={videoLink}
                  onChange={(e) => setVideoLink(e.target.value)}
                  placeholder="https://example.com/video.mp4"
                  style={styles.input}
                  required
                />
              </div>
            </div>

            <div style={styles.formGroup}>
              <label style={styles.label}>Supervisor</label>
              <div style={styles.inputWrapper}>
                <UserIcon style={styles.inputIcon} />
                <select
                  value={supervisor}
                  onChange={(e) => setSupervisor(e.target.value)}
                  style={styles.select}
                >
                  {defaultSupervisors.map((name) => (
                    <option key={name} value={name}>{name}</option>
                  ))}
                  <option value="Other">Other...</option>
                </select>
              </div>
              {supervisor === 'Other' && (
                <input
                  type="text"
                  value={customSupervisor}
                  onChange={(e) => setCustomSupervisor(e.target.value)}
                  placeholder="Enter custom name"
                  style={{ ...styles.input, marginTop: '8px', paddingLeft: '12px' }}
                />
              )}
            </div>

            <div style={styles.formGroup}>
              <label style={styles.label}>Vehicle Number</label>
              <div style={styles.inputWrapper}>
                <TruckIcon style={styles.inputIcon} />
                <select
                  value={vehicle}
                  onChange={(e) => setVehicle(e.target.value)}
                  style={styles.select}
                >
                  {defaultVehicles.map((num) => (
                    <option key={num} value={num}>{num}</option>
                  ))}
                  <option value="Other">Other...</option>
                </select>
              </div>
              {vehicle === 'Other' && (
                <input
                  type="text"
                  value={customVehicle}
                  onChange={(e) => setCustomVehicle(e.target.value)}
                  placeholder="Enter custom number"
                  style={{ ...styles.input, marginTop: '8px', paddingLeft: '12px' }}
                />
              )}
            </div>

            <div style={{ ...styles.formGroup, gridColumn: 'span 2' }}>
              <label style={styles.label}>Model</label>
              <div style={styles.inputWrapper}>
                <RectangleStackIcon style={styles.inputIcon} />
                <select
                  value={model}
                  onChange={(e) => setModel(e.target.value)}
                  style={styles.select}
                >
                  {defaultModels.map((mod) => (
                    <option key={mod} value={mod}>{mod}</option>
                  ))}
                </select>
              </div>
            </div>
          </div>

          <div style={styles.buttonContainer}>
            <button
              onClick={handleStart}
              disabled={isDetecting}
              style={{ ...styles.button, backgroundColor: isDetecting ? '#9CA3AF' : '#22C55E' }}
            >
              <PlayIcon style={{ ...styles.buttonIcon, marginRight: '8px' }} />
              <span>{isDetecting ? 'Detection Running...' : 'Start Detection'}</span>
            </button>
            <button
              onClick={handleStop}
              disabled={!isDetecting}
              style={{ ...styles.button, backgroundColor: !isDetecting ? '#9CA3AF' : '#EF4444' }}
            >
              <StopIcon style={{ ...styles.buttonIcon, marginRight: '8px' }} />
              <span>Stop Detection</span>
            </button>
          </div>
        </div>
      </div>
      <Toast
        message={toastMessage.message}
        type={toastMessage.type}
        onClose={() => setToastMessage({ message: '', type: '' })}
      />
    </div>
  );
}

const styles = {
  appContainer: {
    minHeight: '100vh',
    backgroundImage: `url(${backgroundImage})`,
    backgroundSize: 'cover', // This ensures the image covers the entire container
    backgroundPosition: 'center', // This centers the image
    backgroundAttachment: 'fixed',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    padding: '16px',
    fontFamily: 'sans-serif',
  },
  card: {
    backgroundColor: '#9ec0c9e8',
    padding: '32px',
    borderRadius: '16px',
    boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)',
    maxWidth: '800px',
    width: '100%',
  },
  cardHeader: {
    textAlign: 'center',
    marginBottom: '24px',
  },
  cardHeaderIconContainer: {
    margin: '0 auto 16px',
    width: '64px',
    height: '64px',
    backgroundColor: '#0c0c0eff',
    borderRadius: '16px',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    boxShadow: '0 4px 6px rgba(59, 130, 246, 0.3)',
  },
  cardHeaderIcon: {
    width: '36px',
    height: '36px',
    color: 'white',
  },
  cardTitle: {
    fontSize: '2rem',
    fontWeight: '700',
    color: '#1F2937',
    marginBottom: '8px',
  },
  cardDescription: {
    fontSize: '1rem',
    color: '#6B7280',
  },
  cardContent: {
    display: 'flex',
    flexDirection: 'column',
    gap: '24px',
  },
  formGroup: {
    display: 'flex',
    flexDirection: 'column',
  },
  label: {
    fontSize: '0.875rem',
    fontWeight: '600',
    color: '#4B5563',
    marginBottom: '8px',
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
  },
  inputWrapper: {
    position: 'relative',
  },
  input: {
    display: 'block',
    width: '96%',
    borderRadius: '8px',
    border: '1px solid #090b0eff',
    padding: '12px',
    paddingLeft: '48px',
    color: '#111827',
    outline: 'none',
    transition: 'box-shadow 0.2s, border-color 0.2s',
  },
  inputIcon: {
    position: 'absolute',
    left: '12px',
    top: '50%',
    transform: 'translateY(-50%)',
    height: '24px',
    width: '24px',
    color: '#444447ff',
  },
  select: {
    display: 'block',
    width: '100%',
    borderRadius: '6px',
    border: '1px solid #111214ff',
    padding: '12px',
    paddingLeft: '48px',
    color: '#111827',
    outline: 'none',
    transition: 'box-shadow 0.2s, border-color 0.2s',
    appearance: 'none',
    background: `url('data:image/svg+xml;utf8,<svg fill="none" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20"><path fill="%239CA3AF" d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z"/></svg>') no-repeat right 1rem center / 1.5rem 1.5rem`,
  },
  gridContainer: {
    display: 'grid',
    gridTemplateColumns: '1fr 1fr',
    gap: '24px',
  },
  buttonContainer: {
    marginTop: '32px',
    display: 'flex',
    justifyContent: 'center',
    gap: '16px',
  },
  button: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    gap: '8px',
    padding: '12px 24px',
    borderRadius: '9999px',
    color: '#FFFFFF',
    fontWeight: '600',
    transition: 'transform 0.2s, background-color 0.2s',
    border: 'none',
    cursor: 'pointer',
    boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
  },
  buttonIcon: {
    height: '20px',
    width: '20px',
  },
  statusContainer: {
    marginTop: '32px',
    textAlign: 'center',
    color: '#6B7280',
    fontStyle: 'italic',
  },
  statusMessage: {
    textAlign: 'center',
    color: '#6B7280',
    fontStyle: 'italic',
  },
};