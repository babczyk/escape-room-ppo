using System.Runtime.Serialization.Formatters;
using AI_project_escapeRoom; // Replace 'YourNamespace' with the actual namespace where Game1 is defined
using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Graphics;
using Microsoft.Xna.Framework.Input;
using System.Linq;
class GameEnvironment
{
    private Game1 game;
    private int maxSteps = 500;
    private int currentStep;

    public GameEnvironment(Game1 game)
    {
        this.game = game;
        this.currentStep = 0;
    }

    public double[] GetState()
    {
        // Example state representation
        return new double[]
        {
            game.player.Position.X / game.widthLevel,
            game.player.Position.Y / game.groundLevel,
            game.box.Position.X / game.widthLevel,
            game.box.Position.Y / game.groundLevel,
            game.IsPressed ? 1.0 : 0.0,
            game.IsOpen ? 1.0 : 0.0
        };
    }

    public (double[], int, bool) Step(int action)
    {
        currentStep++;
        int reward = 0;
        bool IsDone = false;

        // Apply action
        switch (action)
        {
            case 0: game.player.Move(new Vector2(-10, 0)); break; // Move left
            case 1: game.player.Move(new Vector2(10, 0)); break;  // Move right
            case 2: if (game.player.IsGrounded) game.player.ApplyForce(new Vector2(0, -250)); break; // Jump
            case 3: game.player.Grab(game.box); break; // Interact (e.g., pick up the box)
            case 4: game.player.DropHeldBox(); break; // Interact (e.g., drop the box)
        }

        // Rewards for key objectives
        if (game.box.Intersects(game.button))
        {
            reward += 100; // Major reward for placing the box on the button
            game.IsPressed = true; // Mark the button as activated
        }

        if (game.player.Intersects(game.box))
        {
            reward += 20; // Reward for successfully picking up or interacting with the box
        }

        if (game.player.Intersects(game.door) && game.IsPressed)
        {
            reward += 200; // High reward for completing the goal
            IsDone = true; // Mark the episode as complete
        }

        // Encouraging progress
        if (game.IsMovingToward(game.box, game.lastPlayerPosition))
        {
            reward += 5; // Encourage moving toward the box
        }

        if (game.IsMovingToward(game.button, game.lastPlayerPosition) && game.player.heldBox != null)
        {
            reward += 10; // Higher reward for moving the box closer to the button
        }

        if (game.IsMovingToward(game.door, game.lastPlayerPosition) && game.IsPressed)
        {
            reward += 10; // Reward for heading toward the door after activating the button
        }

        // Exploration and activity
        if (game.IsExploringNewArea())
        {
            reward += 2; // Small reward for exploring new areas
        }

        if (game.IsIdle())
        {
            reward -= 2; // Slight penalty for standing still to encourage movement
        }

        // Penalties for mistakes
        if (!game.box.Intersects(game.button) && game.player.Intersects(game.door))
        {
            reward -= 50; // Penalty for trying to exit without solving the puzzle
        }

        if (currentStep >= maxSteps)
        {
            reward -= 10; // Increased penalty for exceeding step limit
            ResetPlayerAndBox(); // Reset positions
            IsDone = true; // End the episode
        }

        if (IsOutOfBounds(game.player))
        {
            reward -= 20; // Penalty for going out of bounds
            ResetPlayerPosition(); // Reset player position
            IsDone = true; // End the episode
        }

        if (IsOutOfBounds(game.box))
        {
            reward -= 20; // Penalty for box out of bounds
            ResetBoxPosition(); // Reset box position
            IsDone = true; // End the episode
        }

        return (GetState(), reward, IsDone);
    }

    // Helper methods
    private bool IsOutOfBounds(GameObject obj)
    {
        return obj.Position.X < game.cameraPosition.X || obj.Position.X > game.cameraPosition.X + game.ScreenWidth ||
               obj.Position.Y < game.cameraPosition.Y || obj.Position.Y > game.cameraPosition.Y + game.ScreenHeight;
    }

    private void ResetPlayerAndBox()
    {
        ResetPlayerPosition();
        ResetBoxPosition();
        currentStep = 0;
    }

    private void ResetPlayerPosition()
    {
        game.player.Position = new Vector2(100, game.groundLevel - game.player.Size.Y);
    }

    private void ResetBoxPosition()
    {
        game.box.Position = new Vector2(200, game.groundLevel - game.box.Size.Y);
    }

    public bool IsDone()
    {
        return game.IsOpen || currentStep >= maxSteps;
    }
}