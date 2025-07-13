import React from 'react';
import {Button, Form, Spinner, Card} from 'react-bootstrap';
import { Image as ImageIcon, Wand2 } from "lucide-react";

function LeftUploadCard({ HeaderText, onFileSelect, onUpload, loading, selectedFile, fileInputRef, triggerFileInput, originalImageUrl, modifiedImageUrl, pollinateImage, setShowPollinateModal}) {
    return (
        <Card className="h-100 shadow-sm rounded-4">
            <Card.Header className="text-center bg-dark text-white">
                {HeaderText}
            </Card.Header>
            
            <Card.Body className="cardBodyStyle">

                {/* Hidden file input */}
                <Form>
                    <Form.Control
                        type="file"
                        accept="image/*"
                        onChange={onFileSelect}
                        ref={fileInputRef}
                        className="d-none" 
                    />
                </Form>

                {/* Show uploaded image if available */}
                {pollinateImage ? (
                    <div className="text-center">
                        <img
                        src={pollinateImage}
                        alt="Pollinate AI"
                        className="img-fluid"
                        style={{ objectFit: 'contain' }}
                        />
                    </div>
                ) : originalImageUrl ? (
                    <div className="text-center mb-3">
                        <img
                        src={originalImageUrl}
                        alt="Original Upload"
                        className="img-fluid"
                        style={{ objectFit: 'contain' }}
                        />
                    </div>
                ) : (               
                    // Otherwise show upload icon and prompt
                    <div className="text-center text-muted d-flex flex-column align-items-center">
                        <ImageIcon size={64} className='mb-3' />
                        <Button
                            variant="link"
                            className="p-0 border-0 text-decoration-none" onClick={triggerFileInput} 
                        >
                            <p className="mb-0">
                                Upload an image to begin
                            </p>
                        </Button>

                        <Button
                        className="sparkling-purple-btn w-100 mt-4"
                        onClick={() => setShowPollinateModal(true)}
                        disabled={loading}
                        >
                            <Wand2 size={16} className='mr-2'/>
                            Generate with Pollinate AI
                        </Button>
                    </div>
                )}

                {/* upload Button (only if file selected) */}
                {selectedFile && !modifiedImageUrl && (
                    <div className="mt-3">
                        {loading ? (
                        <Spinner animation="border" size="sm" />) : (
                            <>
                                <div>✔ Upload Completed</div>
                            </>
                        )}
                    </div>
                )}
            </Card.Body>
        </Card>
    );
}


export default LeftUploadCard;