import React from "react"
import {Button, Modal} from "react-bootstrap"

function RestartButton({gameEnded, Loading, onRestart, showConfirm, setShowConfirm }) {
    const handleClick = () => {
        if (!gameEnded)
        {
            setShowConfirm(true);
        }
        else
        {
            onRestart();
        }
    };

    return (
        <>
           <Button 
            variant={gameEnded ? "primary" : "danger"} 
            onClick={handleClick} 
            disabled={Loading}
            >
                {Loading ? "Loading..." : "Restart"}
            </Button>

            <Modal show={showConfirm} onHide={() => setShowConfirm(false)} centered>
                <Modal.Header closeButton>
                    <Modal.Title>
                        Confirm Restart
                    </Modal.Title>
                </Modal.Header>
                
                <Modal.Body>
                    Restarting loses all progress of the current game. <br/>
                    Would you still like to proceed?
                </Modal.Body>

                <Modal.Footer>
                    <Button variant="secondary" onClick={() => setShowConfirm(false)}>
                        Cancel
                    </Button>
                <Button
                    variant="info"
                    onClick={() => {
                    setShowConfirm(false);
                    onRestart();
                    }}
                >
                    Confirm
                </Button>
                </Modal.Footer>
            </Modal> 
        </>
    )
}

export default RestartButton